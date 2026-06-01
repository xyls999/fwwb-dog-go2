# -*- coding: UTF-8 -*-
###########################################################################
# Copyright  1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Critic observation processor."""

import torch

from agent_ppo.conf.conf import WkRuntimeConfig
from tools.base_env.observation_process import ObservationProcess


class CriticObservationProcess(ObservationProcess):
    target_group = "critic"
    _BASE_OBS_DIM = 316

    def _wk_goal_features(self):
        feature_dim = getattr(WkRuntimeConfig.CURRENT, "num_goal_obs", 0)
        if feature_dim <= 0:
            return None

        zeros = torch.zeros(self.env.num_envs, feature_dim, device=self.env.device)
        if feature_dim != 3:
            return zeros

        if hasattr(self, "goal_position_in_robot_frame"):
            self.goal_position_in_robot_frame()
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return zeros

        try:
            robot = self.env.scene["robot"]
            root_pos_w = robot.data.root_pos_w
            quat = robot.data.root_quat_w
        except Exception:
            return zeros

        delta_w = self.env.goal_positions[:, :2] - root_pos_w[:, :2]
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        heading = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        cos_h = torch.cos(-heading)
        sin_h = torch.sin(-heading)
        local_x = cos_h * delta_w[:, 0] - sin_h * delta_w[:, 1]
        local_y = sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1]
        local_goal = torch.stack((local_x, local_y), dim=1)
        local_goal = torch.clamp(local_goal / 10.0, -1.0, 1.0)
        goal_dist = torch.clamp(torch.linalg.norm(delta_w, dim=1), 0.0, 20.0) / 20.0
        return torch.cat((local_goal, goal_dist.unsqueeze(1)), dim=1)

    def process(self):
        obs = self.default_observation()
        if obs.shape[-1] != self._BASE_OBS_DIM:
            raise ValueError(
                f"Critic observation dim mismatch: expected base {self._BASE_OBS_DIM}, got {obs.shape[-1]}."
            )

        goal_features = self._wk_goal_features()
        if goal_features is not None:
            obs = self.concatenate_terms(obs, goal_features)
        return obs
