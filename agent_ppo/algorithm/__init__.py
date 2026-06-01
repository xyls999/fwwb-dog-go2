#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Algorithm package exports for the PPO baseline.

The current baseline only exposes PPO. Additional algorithms can be added
under this package later without changing the public import path.
"""

from .algorithm_ppo import WkPPOTrainer

AlgorithmPPO = WkPPOTrainer

__all__ = [
    "WkPPOTrainer",
    "AlgorithmPPO",
]
