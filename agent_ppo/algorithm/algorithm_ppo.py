#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Legacy PPO training algorithm.

This implementation keeps the previous PPO update behavior while cleaning the
code layout and comments.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn as nn

from agent_ppo.feature.definition import WkRolloutStorage


class WkPPOTrainer:
    """PPO optimizer wrapper used by the legacy locomotion pipeline."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None,
        logger: Any = None,
        monitor: Any = None,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        normalize_value_loss: bool = True,
        num_mini_batches: int = 4,
        num_learning_epochs: int = 5,
        desired_kl: float = 0.01,
        schedule: str = "adaptive",
        **kwargs,
    ):
        """Store PPO hyperparameters and runtime references."""

        self.device = device
        self.actor_critic = model
        self.optimizer = optimizer
        self.logger = logger
        self.monitor = monitor

        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.normalize_value_loss = normalize_value_loss
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.desired_kl = desired_kl
        self.schedule = schedule

        # Note: The legacy training recipe clamps exploration noise after each update
        # Note: so entropy spikes cannot permanently destabilize the gait.
        from agent_ppo.conf.conf import WkRuntimeConfig

        min_std_cfg = getattr(WkRuntimeConfig.CURRENT, "min_normalized_std", None)
        max_std_cfg = getattr(WkRuntimeConfig.CURRENT, "max_normalized_std", None)
        self.min_std = torch.tensor(min_std_cfg, device=device) if min_std_cfg is not None else None
        self.max_std = torch.tensor(max_std_cfg, device=device) if max_std_cfg is not None else None

        self.train_step = 0
        self.last_report_monitor_time = 0.0
        self.storage = None

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: tuple,
        critic_obs_shape: tuple,
        action_shape: tuple,
        device: torch.device = None,
    ):
        """Allocate the rollout storage used by the workflow and PPO update."""

        device = device or self.device
        self.storage = WkRolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=actor_obs_shape,
            privileged_obs_shape=critic_obs_shape,
            actions_shape=action_shape,
            device=device,
        )

    def initialize_rollout_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: tuple,
        critic_obs_shape: tuple,
        action_shape: tuple,
        device: torch.device = None,
    ):
        """Compatibility alias kept for callers that use the newer helper name."""

        self.init_storage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            actor_obs_shape=actor_obs_shape,
            critic_obs_shape=critic_obs_shape,
            action_shape=action_shape,
            device=device,
        )

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor = None) -> tuple:
        """Sample actions and compute rollout values without tracking gradients."""

        if critic_obs is None:
            critic_obs = obs

        with torch.no_grad():
            actions = self.actor_critic.act(obs)
            values = self.actor_critic.evaluate(critic_obs)
            log_probs = self.actor_critic.get_actions_log_prob(actions)
            action_mean = self.actor_critic.action_mean.detach()
            action_std = self.actor_critic.action_std.detach()

        return actions, values, log_probs, action_mean, action_std

    def compute_returns(self, last_obs: torch.Tensor):
        """Finish the rollout by computing GAE returns from the last critic value."""

        with torch.no_grad():
            last_values = self.actor_critic.evaluate(last_obs)

        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def learn(self) -> tuple:
        """Run one PPO optimization round over the internal rollout storage."""

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy_loss = 0.0

        is_recurrent_model = getattr(self.actor_critic, "is_recurrent", False)
        if is_recurrent_model:
            if getattr(self.storage, "saved_hidden_states_a", None) is None:
                raise RuntimeError("Recurrent PPO requires hidden states in rollout storage.")
            batch_generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )
        else:
            batch_generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )

        for sample_index, sample in enumerate(batch_generator):
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hidden_states_batch,
                masks_batch,
            ) = sample

            recurrent_batch = obs_batch.dim() == 3
            if recurrent_batch:
                self.actor_critic.update_distribution(
                    obs_batch,
                    hidden_states=hidden_states_batch,
                    masks=masks_batch,
                )
            else:
                self.actor_critic.update_distribution(obs_batch)

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            entropy_batch = self.actor_critic.entropy

            if recurrent_batch:
                value_batch = self.actor_critic.evaluate(
                    critic_obs_batch.reshape(-1, critic_obs_batch.shape[-1])
                ).view(critic_obs_batch.shape[0], critic_obs_batch.shape[1], 1)
            else:
                value_batch = self.actor_critic.evaluate(critic_obs_batch)

            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            self._wk_update_learning_rate_from_policy_kl(
                mu_batch,
                sigma_batch,
                old_mu_batch,
                old_sigma_batch,
            )

            surrogate_loss = self._wk_compute_clipped_policy_objective(
                actions_log_prob_batch,
                old_actions_log_prob_batch,
                advantages_batch,
            )
            value_loss = self._wk_compute_clipped_value_loss(
                value_batch,
                returns_batch,
                target_values_batch,
            )
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            if not torch.isfinite(loss):
                if self.logger is not None:
                    self.logger.warning(
                        f"[PPO] NaN/Inf loss detected at step {self.train_step}, "
                        f"mini-batch {sample_index}. Skipping this update. "
                        f"surrogate={surrogate_loss.item()}, value={value_loss.item()}"
                    )
                continue

            self.optimizer.zero_grad()
            loss.backward()

            gradients_are_finite = True
            for param in self.actor_critic.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    gradients_are_finite = False
                    break

            if not gradients_are_finite:
                if self.logger is not None:
                    self.logger.warning(
                        f"[PPO] NaN/Inf gradient detected at step {self.train_step}, "
                        f"mini-batch {sample_index}. Zeroing grads and skipping optimizer.step()."
                    )
                self.optimizer.zero_grad()
                continue

            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._wk_clamp_policy_std_after_update()

            surrogate_loss_value = surrogate_loss.item()
            value_loss_value = value_loss.item()
            entropy_loss_value = entropy_batch.mean().item()
            mean_surrogate_loss += surrogate_loss_value if not (surrogate_loss_value != surrogate_loss_value) else 0.0
            mean_value_loss += value_loss_value if not (value_loss_value != value_loss_value) else 0.0
            mean_entropy_loss += entropy_loss_value if not (entropy_loss_value != entropy_loss_value) else 0.0

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates

        self._wk_report_ppo_training_metrics(
            mean_surrogate_loss,
            mean_value_loss,
            mean_entropy_loss,
        )

        self.train_step += 1
        return mean_surrogate_loss, mean_value_loss, mean_entropy_loss

    def _wk_clamp_policy_std_after_update(self) -> None:
        """Clamp policy exploration std after each update using configured bounds."""

        if not hasattr(self.actor_critic, "std") or self.min_std is None:
            return

        min_std = self.min_std
        max_std = self.max_std
        if min_std.shape != self.actor_critic.std.data.shape:
            min_std = torch.zeros_like(self.actor_critic.std.data)
        if max_std is None or max_std.shape != self.actor_critic.std.data.shape:
            max_std = torch.full_like(self.actor_critic.std.data, 1.0e6)

        safe_std = torch.nan_to_num(
            self.actor_critic.std.data,
            nan=1.0,
            posinf=float(torch.max(max_std).item()),
            neginf=0.0,
        )
        self.actor_critic.std.data.copy_(torch.clamp(safe_std, min=min_std, max=max_std))

    def _wk_update_learning_rate_from_policy_kl(
        self,
        mu_batch: torch.Tensor,
        sigma_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
    ):
        """Adapt the learning rate when policy drift deviates from the target KL."""

        if self.desired_kl is None or self.schedule != "adaptive":
            return

        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)

            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _wk_compute_clipped_policy_objective(
        self,
        actions_log_prob_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the clipped PPO surrogate objective for the actor."""

        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio,
            1.0 - self.clip_param,
            1.0 + self.clip_param,
        )
        return torch.max(surrogate, surrogate_clipped).mean()

    def _wk_compute_clipped_value_loss(
        self,
        value_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        target_values_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the critic regression loss with clipping and variance normalization."""

        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param,
                self.clip_param,
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            raw_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            raw_loss = (returns_batch - value_batch).pow(2).mean()

        if self.normalize_value_loss:
            returns_var = returns_batch.detach().var() + 1e-8
            return raw_loss / returns_var

        return raw_loss

    def _wk_report_ppo_training_metrics(
        self,
        mean_surrogate_loss: float,
        mean_value_loss: float,
        mean_entropy_loss: float,
    ):
        """Send coarse PPO training health metrics to the monitor once per minute."""

        now = time.time()
        if now - self.last_report_monitor_time < 60:
            return

        monitor_data = {
            "policy_loss": mean_surrogate_loss,
            "value_loss": mean_value_loss,
            "entropy_loss": mean_entropy_loss,
            "total_loss": mean_surrogate_loss + mean_value_loss + mean_entropy_loss,
            "learning_rate": self.learning_rate,
        }
        if self.monitor is not None:
            self.monitor.put_data({os.getpid(): monitor_data})

        self.last_report_monitor_time = now


AlgorithmPPO = WkPPOTrainer
