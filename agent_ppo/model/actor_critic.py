#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feed-forward actor-critic network used by the legacy PPO pipeline.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal


def wk_build_activation_module(activation: str) -> nn.Module:
    """Return the activation module requested by the stage configuration."""

    activation_map = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if activation not in activation_map:
        raise ValueError(f"Unknown activation: {activation}. Available: {list(activation_map.keys())}")
    return activation_map[activation]


def wk_resolve_nn_activation(activation: str) -> nn.Module:
    """Compatibility alias for older imports."""

    return wk_build_activation_module(activation)


class WkActorCritic(nn.Module):
    """Actor-critic network with flat actor and critic observation tensors."""

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int] | list[int] = (512, 256, 128),
        activation: str = "elu",
        rnn_type: str = "lstm",
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs: dict[str, Any],
    ) -> None:
        """Build the actor, critic, and action distribution parameterization."""

        super().__init__()

        activation_module = wk_build_activation_module(activation)

        # Note: The constructor still accepts recurrent arguments so the external
        # Note: interface stays steady, but the model itself remains feed-forward.
        self._hidden_states = None

        actor_layers = [nn.Linear(num_obs, actor_hidden_dims[0]), activation_module]
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1])
                )
                actor_layers.append(activation_module)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), activation_module]
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(
                    nn.Linear(
                        critic_hidden_dims[layer_index],
                        critic_hidden_dims[layer_index + 1],
                    )
                )
                critic_layers.append(nn.LayerNorm(critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation_module)
        self.critic = nn.Sequential(*critic_layers)

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(
                f"Unknown noise_std_type: {noise_std_type}. Should be 'scalar' or 'log'"
            )

        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def wk_init_weights(sequential, scales):
        """Orthogonally initialize each linear layer in a sequential module."""

        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[index])
            for index, module in enumerate(
                module for module in sequential if isinstance(module, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        """Compatibility hook for recurrent models."""

        pass

    def get_hidden_states(self):
        """Compatibility hook for recurrent models."""

        return None

    def set_hidden_states(self, hidden_states):
        """Compatibility hook for recurrent models."""

        self._hidden_states = None

    def forward(self):
        """The legacy interface uses act/evaluate instead of forward()."""

        raise NotImplementedError

    @property
    def action_mean(self):
        """Expose the current policy mean after update_distribution()."""

        return self.distribution.mean

    @property
    def action_std(self):
        """Expose the current policy standard deviation."""

        return self.distribution.stddev

    @property
    def entropy(self):
        """Return the summed action entropy per environment."""

        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(
        self,
        obs: torch.Tensor,
        hidden_states=None,
        masks: torch.Tensor | None = None,
    ):
        """Build the action distribution from a batch of actor observations."""

        if obs.dim() != 2:
            raise ValueError(f"Actor observation must be 2D [B, num_obs], got shape {tuple(obs.shape)}")

        action_mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            action_std = self.std.clamp(min=1e-6).expand_as(action_mean)
        elif self.noise_std_type == "log":
            action_std = torch.exp(self.log_std).expand_as(action_mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")

        self.distribution = Normal(action_mean, action_std)

    def act(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample stochastic actions for rollout collection."""

        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic mean actions for evaluation."""

        return self.actor(obs)

    def evaluate(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Estimate state values from critic observations."""

        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return the log probability of sampled actions under the current policy."""

        return self.distribution.log_prob(actions).sum(dim=-1)


ActorCritic = WkActorCritic
build_activation_module = wk_build_activation_module
resolve_nn_activation = wk_resolve_nn_activation
