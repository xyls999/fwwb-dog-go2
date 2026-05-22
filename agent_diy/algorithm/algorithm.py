#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
PPO algorithm — aligned with rsl_rl PPO.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any
import time
import os


class Algorithm:
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
        num_mini_batches: int = 4,
        num_learning_epochs: int = 5,
        desired_kl: float = 0.01,
        schedule: str = "adaptive",
        **kwargs,
    ):
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
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.desired_kl = desired_kl
        self.schedule = schedule

        from agent_diy.conf.conf import Config
        self.min_std = torch.tensor(Config.CURRENT.min_normalized_std, device=device)

        self.train_step = 0
        self.last_report_monitor_time = 0
        self.storage = None

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, device=None):
        from agent_ppo.feature.definition import RolloutStorage
        self.storage = RolloutStorage(
            num_envs=num_envs, num_transitions_per_env=num_transitions_per_env,
            obs_shape=actor_obs_shape, privileged_obs_shape=critic_obs_shape,
            actions_shape=action_shape, device=device or self.device,
        )

    def act(self, obs, critic_obs=None):
        if critic_obs is None:
            critic_obs = obs
        with torch.no_grad():
            actions = self.actor_critic.act(obs)
            values = self.actor_critic.evaluate(critic_obs)
            log_probs = self.actor_critic.get_actions_log_prob(actions)
            action_mean = self.actor_critic.action_mean.detach()
            action_std = self.actor_critic.action_std.detach()
        return actions, values, log_probs, action_mean, action_std

    def compute_returns(self, last_obs):
        with torch.no_grad():
            last_values = self.actor_critic.evaluate(last_obs)
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def learn(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for sample_idx, sample in enumerate(generator):
            (
                obs_batch, critic_obs_batch, actions_batch, target_values_batch,
                advantages_batch, returns_batch, old_actions_log_prob_batch,
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch,
            ) = sample

            self.actor_critic.update_distribution(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            entropy_batch = self.actor_critic.entropy
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std

            self._update_learning_rate(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if not torch.isfinite(loss):
                if self.logger:
                    self.logger.warning(f"[PPO] NaN loss at step {self.train_step}, mini-batch {sample_idx}")
                continue

            self.optimizer.zero_grad()
            loss.backward()

            grad_ok = all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in self.actor_critic.parameters()
            )
            if not grad_ok:
                if self.logger:
                    self.logger.warning(f"[PPO] NaN gradient at step {self.train_step}")
                self.optimizer.zero_grad()
                continue

            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Clamp std: enforce [min_std, 1e6], avoid mixed Tensor/float in clamp()
            if hasattr(self.actor_critic, "std") and self.min_std is not None:
                safe = torch.nan_to_num(self.actor_critic.std.data, nan=1.0, posinf=1e6, neginf=0.0)
                safe = torch.maximum(safe, self.min_std)
                safe = torch.clamp(safe, max=1e6)
                self.actor_critic.std.data.copy_(safe)

            mean_surrogate_loss += surrogate_loss.item()
            mean_value_loss += value_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()

        n = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= n
        mean_surrogate_loss /= n
        mean_entropy_loss /= n

        self._report(mean_surrogate_loss, mean_value_loss, mean_entropy_loss)
        self.train_step += 1
        return mean_surrogate_loss, mean_value_loss, mean_entropy_loss

    def _update_learning_rate(self, mu, sigma, old_mu, old_sigma):
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma / old_sigma + 1e-5)
                + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2 * sigma.pow(2))
                - 0.5, dim=-1,
            )
            kl_mean = torch.mean(kl)
            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.learning_rate

    def _report(self, sl, vl, el):
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            if self.monitor:
                self.monitor.put_data({os.getpid(): {
                    "policy_loss": sl, "value_loss": vl, "entropy_loss": el,
                    "total_loss": sl + vl + el, "learning_rate": self.learning_rate,
                }})
            self.last_report_monitor_time = now
