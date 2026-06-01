#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Model package exports for the PPO baseline.

Only the flat actor-critic network is required today. Keeping the exports in
one place makes it easier to add terrain encoders or vision backbones later
without changing downstream import sites.
"""

from agent_ppo.model.actor_critic import WkActorCritic, wk_resolve_nn_activation

ActorCritic = WkActorCritic
resolve_nn_activation = wk_resolve_nn_activation

__all__ = [
    "WkActorCritic",
    "wk_resolve_nn_activation",
    "ActorCritic",
    "resolve_nn_activation",
]
