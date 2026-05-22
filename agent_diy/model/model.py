#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Actor-Critic for residual action prediction, aligned with rsl_rl GaussianDistribution.
RL outputs residual in [-residual_clip, residual_clip], added to CPG base.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Any


def resolve_activation(activation: str) -> nn.Module:
    m = {"elu": nn.ELU(), "selu": nn.SELU(), "relu": nn.ReLU(),
         "lrelu": nn.LeakyReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}
    return m[activation]


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: tuple = (512, 256, 128),
        critic_hidden_dims: tuple = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        residual_clip: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        act_fn = resolve_activation(activation)
        self.residual_clip = residual_clip

        # Actor: obs → [512,256,128] → Tanh → *clip
        layers = []
        layers.append(nn.Linear(num_obs, actor_hidden_dims[0]))
        layers.append(act_fn)
        for i in range(len(actor_hidden_dims) - 1):
            layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            layers.append(act_fn)
        layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        layers.append(nn.Tanh())
        self.actor = nn.Sequential(*layers)

        # Critic: critic_obs → [512,256,128] → 1
        c_layers = []
        c_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        c_layers.append(act_fn)
        for i in range(len(critic_hidden_dims) - 1):
            c_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            c_layers.append(nn.LayerNorm(critic_hidden_dims[i + 1]))
            c_layers.append(act_fn)
        c_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*c_layers)

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        else:
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None): pass
    def forward(self): raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean if self.distribution is not None else None
    @property
    def action_std(self):
        return self.distribution.stddev if self.distribution is not None else None
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1) if self.distribution is not None else None

    def update_distribution(self, obs: torch.Tensor):
        mean = self.actor(obs) * self.residual_clip
        if self.noise_std_type == "scalar":
            std = self.std.clamp(min=1e-6).expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs) * self.residual_clip

    def evaluate(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)
