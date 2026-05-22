#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Data definitions for the hybrid CPG + RL agent.
"""

from common_python.utils.common_func import create_cls, Frame
import torch
import numpy as np
from agent_diy.conf.conf import Config


ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls(
    "ActData",
    action=None,
    cpg_action=None,
    rl_residual=None,
)


def sample_process(collector):
    return collector.sample_process()


def build_frame(frame_no, obs, actions, dones, rewards):
    return Frame(
        frame_no=frame_no,
        obs=obs,
        actions=actions,
        done=dones,
        rewards=rewards,
    )
