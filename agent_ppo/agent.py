#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PPO agent entrypoint.

This file keeps the previous PPO agent behavior while cleaning naming,
comments, and checkpoint compatibility details.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.optim as optim

from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm_ppo import WkPPOTrainer
from agent_ppo.conf.conf import WkRuntimeConfig, _load_toml
from agent_ppo.feature.definition import ActData
from agent_ppo.model.actor_critic import WkActorCritic
from tools.train_env_conf_validate import check_usr_conf


def wk_seed_global_random_generators(seed: int = 0) -> None:
    """Keep startup deterministic across training and evaluation restarts."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


wk_seed_global_random_generators(0)


def wk_reshape_height_scan_grid(
    observation_tensor: torch.Tensor,
    scan_start: int = 45,
    scan_size: int = 256,
):
    """Reshape the flattened height-scan slice into a square grid when possible."""

    if (
        observation_tensor is None
        or not hasattr(observation_tensor, "shape")
        or observation_tensor.shape[-1] < scan_start + scan_size
    ):
        return None

    if observation_tensor.dim() == 1:
        observation_tensor = observation_tensor.unsqueeze(0)

    side_length = int(scan_size**0.5)
    if side_length * side_length != scan_size:
        return None

    return observation_tensor[:, scan_start : scan_start + scan_size].view(
        observation_tensor.shape[0],
        side_length,
        side_length,
    )


def wk_infer_pre_maze_terrain_class(observation_tensor, rl_navigation_conf):
    """Infer whether the upcoming non-maze terrain looks flat, sloped, or stair-like."""

    height_grid = wk_reshape_height_scan_grid(
        observation_tensor,
        scan_start=int(rl_navigation_conf.get("scan_start", 45)),
        scan_size=int(rl_navigation_conf.get("scan_size", 256)),
    )
    if height_grid is None:
        return None

    row_start = max(int(rl_navigation_conf.get("terrain_row_start", 3)), 0)
    row_end = min(int(rl_navigation_conf.get("terrain_row_end", 13)), height_grid.shape[1])
    front_col_count = min(int(rl_navigation_conf.get("terrain_front_cols", 8)), height_grid.shape[2])
    if row_end <= row_start or front_col_count <= 1:
        return None

    front_sector = height_grid[:, row_start:row_end, :front_col_count]
    if front_sector.shape[1] == 0 or front_sector.shape[2] <= 1:
        return None

    lateral_std = front_sector.std(dim=1, unbiased=False).mean(dim=1)
    lateral_deltas = front_sector[:, :, 1:] - front_sector[:, :, :-1]
    abs_lateral_deltas = lateral_deltas.abs()
    if abs_lateral_deltas.numel() == 0:
        return None

    terrain_quantile = float(rl_navigation_conf.get("terrain_step_quantile", 0.85))
    terrain_quantile = min(max(terrain_quantile, 0.0), 1.0)
    step_strength = torch.quantile(abs_lateral_deltas.flatten(1), terrain_quantile, dim=1)
    sign_consistency = lateral_deltas.mean(dim=(1, 2)).abs() / (
        abs_lateral_deltas.mean(dim=(1, 2)) + 1e-6
    )
    if lateral_deltas.shape[2] > 1:
        second_difference = (
            lateral_deltas[:, :, 1:] - lateral_deltas[:, :, :-1]
        ).abs().mean(dim=(1, 2))
    else:
        second_difference = torch.zeros(
            observation_tensor.shape[0],
            device=observation_tensor.device,
            dtype=observation_tensor.dtype,
        )

    is_uniform_surface = lateral_std < float(
        rl_navigation_conf.get("terrain_lateral_std_threshold", 0.18)
    )
    is_not_wall = front_sector.amin(dim=(1, 2)) > float(
        rl_navigation_conf.get("terrain_wall_height_threshold", -1.05)
    )
    looks_like_terrain = is_uniform_surface & is_not_wall & (
        step_strength > float(rl_navigation_conf.get("terrain_slope_delta_threshold", 0.035))
    )
    looks_like_stairs = looks_like_terrain & (
        (step_strength > float(rl_navigation_conf.get("terrain_stair_delta_threshold", 0.10)))
        | (
            second_difference
            > float(rl_navigation_conf.get("terrain_stair_second_diff_threshold", 0.055))
        )
    )
    looks_like_slope = looks_like_terrain & ~looks_like_stairs & (
        sign_consistency
        > float(rl_navigation_conf.get("terrain_slope_sign_consistency_threshold", 0.55))
    )

    terrain_id = torch.zeros(
        observation_tensor.shape[0],
        dtype=torch.long,
        device=observation_tensor.device,
    )
    terrain_id = torch.where(looks_like_slope, torch.ones_like(terrain_id), terrain_id)
    terrain_id = torch.where(looks_like_stairs, torch.full_like(terrain_id, 2), terrain_id)
    return terrain_id


class Agent(BaseAgent):
    """KaiwuDRL-facing PPO agent wrapper."""

    def __init__(self, agent_type="player", device="cuda", logger=None, monitor=None):
        self.cur_model_name = "WkActorCritic"
        self.device = device
        self.logger = logger
        self.monitor = monitor

        runtime_env_conf, runtime_env_conf_path, is_eval_mode, stage_config = WkRuntimeConfig.load_conf(
            self.logger
        )
        self.is_eval = is_eval_mode
        is_valid_conf, validation_message = check_usr_conf(
            runtime_env_conf,
            is_eval_mode,
            self.logger,
        )
        if not is_valid_conf:
            error_message = (
                f"check_usr_conf is {is_valid_conf}, message is {validation_message}, "
                f"please check {runtime_env_conf_path}"
            )
            if self.logger is not None:
                self.logger.error(error_message)
            raise Exception(error_message)

        self.stage = stage_config
        self.num_envs = runtime_env_conf["env"]["num_envs"]

        # Note: These dimensions are architecture constants bound to StageConfig.
        # Note: Keeping them out of TOML avoids accidental checkpoint incompatibility.
        self.num_actions = stage_config.num_actions
        self.num_critic_obs = stage_config.num_critic_observations

        num_proprio_obs = stage_config.num_proprio_obs
        num_scan_obs = stage_config.num_scan
        num_goal_obs = getattr(stage_config, "num_goal_obs", 0)
        self.num_obs = num_proprio_obs + num_scan_obs + num_goal_obs

        self._wk_build_flat_actor_critic_components(stage_config)

        train_stage_runtime_conf = {}
        if is_eval_mode and stage_config.task_type == "track":
            train_stage_runtime_conf = self._wk_load_train_stage_runtime_config(stage_config)

        # Note: Evaluation retains the policy command input observation aligned with the
        # Note: track-navigation objective even if the platform command ranges differ.
        self.eval_command_override = None
        self.eval_phase_command_enabled = False
        self.eval_pre_maze_command = None
        self.eval_slope_command = None
        self.eval_stairs_command = None
        self.eval_maze_command = None
        self.eval_phase_maze_goal_dist_gate = 14.0
        self.eval_rl_nav_conf = {}
        if is_eval_mode and stage_config.task_type == "track":
            rl_navigation_conf = train_stage_runtime_conf.get("rl_navigation", {}).copy()
            rl_navigation_conf.update(runtime_env_conf.get("rl_navigation", {}))
            self.eval_rl_nav_conf = rl_navigation_conf.copy()
            self.eval_phase_command_enabled = bool(
                rl_navigation_conf.get("phase_command_enabled", False)
            )
            self.eval_phase_maze_goal_dist_gate = float(
                rl_navigation_conf.get("phase_maze_goal_dist_gate", 14.0)
            )
            pre_maze_range = rl_navigation_conf.get("pre_maze_lin_vel_x", [0.75, 1.0])
            slope_range = rl_navigation_conf.get("slope_lin_vel_x", pre_maze_range)
            stairs_range = rl_navigation_conf.get("stairs_lin_vel_x", pre_maze_range)
            maze_range = rl_navigation_conf.get("maze_lin_vel_x", [0.45, 0.65])
            if len(pre_maze_range) == 2:
                self.eval_pre_maze_command = torch.tensor(
                    [0.5 * (float(pre_maze_range[0]) + float(pre_maze_range[1])), 0.0, 0.0],
                    device=self.device,
                    dtype=torch.float32,
                )
            if len(slope_range) == 2:
                self.eval_slope_command = torch.tensor(
                    [0.5 * (float(slope_range[0]) + float(slope_range[1])), 0.0, 0.0],
                    device=self.device,
                    dtype=torch.float32,
                )
            if len(stairs_range) == 2:
                self.eval_stairs_command = torch.tensor(
                    [0.5 * (float(stairs_range[0]) + float(stairs_range[1])), 0.0, 0.0],
                    device=self.device,
                    dtype=torch.float32,
                )
            if len(maze_range) == 2:
                self.eval_maze_command = torch.tensor(
                    [0.5 * (float(maze_range[0]) + float(maze_range[1])), 0.0, 0.0],
                    device=self.device,
                    dtype=torch.float32,
                )
            if bool(rl_navigation_conf.get("eval_command_override", True)):
                eval_command = rl_navigation_conf.get("eval_command", [0.55, 0.0, 0.0])
                if len(eval_command) == 3:
                    self.eval_command_override = torch.tensor(
                        eval_command,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    if self.logger is not None:
                        self.logger.info(
                            "[RLNavigation] Eval policy command obs override enabled: "
                            f"{eval_command}"
                        )

        self.num_steps_per_env = stage_config.num_steps_per_env
        self.save_interval = stage_config.model_save_interval

        self.algorithm.init_storage(
            self.num_envs,
            self.num_steps_per_env,
            actor_obs_shape=(self.num_obs,),
            critic_obs_shape=(self.num_critic_obs,),
            action_shape=(self.num_actions,),
            device=self.device,
        )

        super().__init__(agent_type, device, logger, monitor)

    def _wk_load_train_stage_runtime_config(self, stage_config):
        """Load the train-stage TOML so evaluation can reuse RL-navigation settings."""

        train_stage_config_file = (
            f"agent_ppo/conf/train_env_conf_{stage_config.task_type}_{stage_config.name}.toml"
        )
        if not os.path.exists(train_stage_config_file):
            return {}
        try:
            return _load_toml(train_stage_config_file)
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning(
                    "[RLNavigation] Failed to load train stage config "
                    f"from {train_stage_config_file}: {exc}"
                )
            return {}

    def _wk_inject_eval_command_observation(self, policy_obs):
        """Inject evaluation-time command anchors into policy observations.

        Evaluation sometimes dispatches a single observation vector with shape
        ``(obs_dim,)`` instead of a batch with shape ``(1, obs_dim)``. This
        helper normalizes both cases to a temporary batch view so the same
        command-patching logic works in train and eval code paths.
        """

        if policy_obs is None or policy_obs.shape[-1] < 9:
            return policy_obs

        input_was_single_observation = policy_obs.dim() == 1
        if input_was_single_observation:
            policy_obs = policy_obs.unsqueeze(0)

        if (
            self.eval_phase_command_enabled
            and self.eval_pre_maze_command is not None
            and self.eval_maze_command is not None
            and policy_obs.shape[-1] >= 304
        ):
            eval_policy_obs = policy_obs.clone()
            goal_distance = torch.clamp(eval_policy_obs[:, 303], 0.0, 1.0) * 20.0
            maze_phase = goal_distance < self.eval_phase_maze_goal_dist_gate
            command = self.eval_pre_maze_command.to(
                device=policy_obs.device,
                dtype=policy_obs.dtype,
            ).expand(policy_obs.shape[0], -1)
            if bool(self.eval_rl_nav_conf.get("terrain_phase_speed_enabled", False)):
                terrain_id = wk_infer_pre_maze_terrain_class(
                    eval_policy_obs,
                    self.eval_rl_nav_conf,
                )
                if terrain_id is not None:
                    if self.eval_slope_command is not None:
                        slope_command = self.eval_slope_command.to(
                            device=policy_obs.device,
                            dtype=policy_obs.dtype,
                        ).expand(policy_obs.shape[0], -1)
                        command = torch.where((terrain_id == 1).unsqueeze(1), slope_command, command)
                    if self.eval_stairs_command is not None:
                        stairs_command = self.eval_stairs_command.to(
                            device=policy_obs.device,
                            dtype=policy_obs.dtype,
                        ).expand(policy_obs.shape[0], -1)
                        command = torch.where((terrain_id == 2).unsqueeze(1), stairs_command, command)
            maze_command = self.eval_maze_command.to(
                device=policy_obs.device,
                dtype=policy_obs.dtype,
            ).expand(policy_obs.shape[0], -1)
            command = torch.where(maze_phase.unsqueeze(1), maze_command, command)
            eval_policy_obs[:, 6:9] = command
            return eval_policy_obs.squeeze(0) if input_was_single_observation else eval_policy_obs

        if self.eval_command_override is None:
            return policy_obs.squeeze(0) if input_was_single_observation else policy_obs

        eval_policy_obs = policy_obs.clone()
        eval_policy_obs[:, 6:9] = self.eval_command_override.to(
            device=policy_obs.device,
            dtype=policy_obs.dtype,
        )
        return eval_policy_obs.squeeze(0) if input_was_single_observation else eval_policy_obs

    def _wk_build_flat_actor_critic_components(self, stage_config):
        """Build the feed-forward actor-critic and PPO optimizer used by legacy PPO."""

        self.model = WkActorCritic(
            num_obs=self.num_obs,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.num_actions,
            actor_hidden_dims=stage_config.actor_hidden_dims,
            critic_hidden_dims=stage_config.critic_hidden_dims,
            activation=stage_config.activation,
            init_noise_std=getattr(stage_config, "init_noise_std", 1.0),
        ).to(self.device)

        if self.logger is not None:
            self.logger.info(f"Actor MLP: {self.model.actor}")
            self.logger.info(f"Critic MLP: {self.model.critic}")

        optimizer_params = [{"params": self.model.parameters(), "name": "actor_critic"}]
        self.optimizer = optim.Adam(optimizer_params, lr=stage_config.lr)

        self.algorithm = WkPPOTrainer(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            logger=self.logger,
            monitor=self.monitor,
            learning_rate=stage_config.lr,
            clip_param=getattr(stage_config, "clip_param", 0.2),
            entropy_coef=getattr(stage_config, "entropy_coef", 0.01),
            desired_kl=getattr(stage_config, "desired_kl", 0.01),
            num_mini_batches=stage_config.num_mini_batches,
            num_learning_epochs=stage_config.num_learning_epochs,
        )

    def exploit(self, list_obs_data):
        """Run deterministic policy inference for evaluation.

        The original evaluation path forwards the full policy observation tensor
        directly into ``exploit``. Keep that behavior so vectorized evaluation
        still receives one action row per environment. A single observation
        vector is still accepted and promoted to batch form for inference.
        """

        policy_obs = list_obs_data
        if not torch.is_tensor(policy_obs):
            if isinstance(policy_obs, (list, tuple)) and len(policy_obs) == 1:
                policy_obs = policy_obs[0]
            else:
                raise TypeError(
                    "exploit expects a policy observation tensor or a single-item container"
                )
        if policy_obs.dim() == 1:
            policy_obs = policy_obs.unsqueeze(0)
        with torch.no_grad():
            eval_policy_obs = self._wk_inject_eval_command_observation(policy_obs)
            actions = self.algorithm.actor_critic.act_inference(eval_policy_obs)
            # Note: Retain the control action batch dimension steady for the Isaac control action manager.
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
            return [ActData(action=actions)]

    def learn(self, list_sample_data=None):
        """Trigger PPO optimization using rollout storage filled by the workflow."""

        return self.algorithm.learn()

    def predict(self, list_obs_data):
        """Run stochastic policy inference for rollout collection."""

        policy_obs = list_obs_data[0]
        critic_obs = list_obs_data[1]

        with torch.no_grad():
            if self.is_eval:
                policy_obs = self._wk_inject_eval_command_observation(policy_obs)

            hidden_states = None
            if getattr(self.algorithm.actor_critic, "is_recurrent", False):
                current_hidden_states = self.algorithm.actor_critic.get_hidden_states()
                if (
                    current_hidden_states is None
                    or current_hidden_states[0].shape[1] != policy_obs.shape[0]
                ):
                    self.algorithm.actor_critic._init_hidden_states(
                        policy_obs.shape[0],
                        policy_obs.device,
                        policy_obs.dtype,
                    )
                    current_hidden_states = self.algorithm.actor_critic.get_hidden_states()
                hidden_states = tuple(
                    hidden_state.detach().clone() for hidden_state in current_hidden_states
                )

            actions = self.algorithm.actor_critic.act(policy_obs)
            values = self.algorithm.actor_critic.evaluate(critic_obs)
            log_probs = self.algorithm.actor_critic.get_actions_log_prob(actions)
            action_mean = self.algorithm.actor_critic.action_mean.detach()
            action_std = self.algorithm.actor_critic.action_std.detach()

            return (
                actions,
                values,
                log_probs,
                action_mean,
                action_std,
                policy_obs.detach(),
                critic_obs.detach(),
                hidden_states,
            )

    def save_model(self, path=None, id="1", *args, **kwargs):
        """Save the current model checkpoint without changing legacy path layout."""

        save_directory = path or os.path.join(os.path.dirname(__file__), "checkpoints")
        os.makedirs(save_directory, exist_ok=True)
        model_file_path = f"{save_directory}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        if self.logger is not None:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1", *args, **kwargs):
        """Load a checkpoint, falling back to partial transfer when shapes differ."""

        load_directory = path or os.path.join(os.path.dirname(__file__), "checkpoints")
        model_file_path = f"{load_directory}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            if self.logger is not None:
                self.logger.info(f"current model is {model_file_path}, so skip load model")
            return

        pretrained_state = torch.load(model_file_path, map_location=self.device)
        current_state = self.model.state_dict()

        has_shape_mismatch = False
        for param_name, pretrained_param in pretrained_state.items():
            if param_name in current_state and pretrained_param.shape != current_state[param_name].shape:
                has_shape_mismatch = True
                break

        if not has_shape_mismatch:
            self.model.load_state_dict(pretrained_state)
            if self.logger is not None:
                self.logger.info(f"load model {model_file_path} successfully (exact match)")
        else:
            self._wk_transfer_checkpoint_parameters_partially(
                self.model,
                pretrained_state,
                model_file_path,
            )

        self._wk_clamp_loaded_action_std_parameters()
        self.cur_model_name = model_file_path

    def _wk_clamp_loaded_action_std_parameters(self):
        """Clamp loaded exploration parameters back into the configured safe range."""

        min_std_cfg = getattr(self.stage, "min_normalized_std", None)
        max_std_cfg = getattr(self.stage, "max_normalized_std", None)
        if min_std_cfg is None and max_std_cfg is None:
            return

        def wk_bound_std(std_tensor):
            posinf_value = 1.0e6
            if max_std_cfg is not None:
                max_std = torch.tensor(max_std_cfg, device=self.device, dtype=std_tensor.dtype)
                if max_std.shape == std_tensor.data.shape:
                    posinf_value = float(torch.max(max_std).item())
            bounded_std = torch.nan_to_num(
                std_tensor.data,
                nan=1.0,
                posinf=posinf_value,
                neginf=0.0,
            )
            if min_std_cfg is not None:
                min_std = torch.tensor(min_std_cfg, device=self.device, dtype=std_tensor.dtype)
                if min_std.shape == bounded_std.shape:
                    bounded_std = torch.maximum(bounded_std, min_std)
            if max_std_cfg is not None:
                max_std = torch.tensor(max_std_cfg, device=self.device, dtype=std_tensor.dtype)
                if max_std.shape == bounded_std.shape:
                    bounded_std = torch.minimum(bounded_std, max_std)
            std_tensor.data.copy_(bounded_std)

        with torch.no_grad():
            if hasattr(self.model, "std"):
                wk_bound_std(self.model.std)
                if self.logger is not None:
                    self.logger.info(
                        f"[PPO] action std bounds enforced: min={min_std_cfg}, max={max_std_cfg}"
                    )
            elif hasattr(self.model, "log_std"):
                log_std = torch.nan_to_num(
                    self.model.log_std.data,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                if min_std_cfg is not None:
                    min_std = torch.tensor(min_std_cfg, device=self.device, dtype=log_std.dtype)
                    if min_std.shape == log_std.shape:
                        log_std = torch.maximum(log_std, torch.log(min_std))
                if max_std_cfg is not None:
                    max_std = torch.tensor(max_std_cfg, device=self.device, dtype=log_std.dtype)
                    if max_std.shape == log_std.shape:
                        log_std = torch.minimum(log_std, torch.log(max_std))
                self.model.log_std.data.copy_(log_std)
                if self.logger is not None:
                    self.logger.info(
                        f"[PPO] log action std bounds enforced: min={min_std_cfg}, max={max_std_cfg}"
                    )

    def _wk_transfer_checkpoint_parameters_partially(self, model, pretrained_state, model_file_path):
        """Transfer checkpoint weights into the current model when tensor shapes differ."""

        current_state = model.state_dict()
        loaded_keys = []
        partial_keys = []
        skipped_keys = []

        for param_name in current_state:
            if param_name not in pretrained_state:
                skipped_keys.append(param_name)
                continue

            if getattr(model, "is_recurrent", False) and param_name in {
                "actor.0.weight",
                "actor.0.bias",
            }:
                skipped_keys.append(f"{param_name} (recurrent front-end)")
                continue

            old_param = pretrained_state[param_name]
            new_param = current_state[param_name]

            if old_param.shape == new_param.shape:
                new_param.copy_(old_param)
                loaded_keys.append(param_name)
                continue

            with torch.no_grad():
                new_param.zero_()
                copy_slices = tuple(
                    slice(0, min(old_size, new_size))
                    for old_size, new_size in zip(old_param.shape, new_param.shape)
                )
                if param_name == "actor.0.weight":
                    base_feature_dim = self.stage.num_proprio_obs + self.stage.num_scan
                    copy_slices = (
                        slice(0, min(old_param.shape[0], new_param.shape[0])),
                        slice(0, min(base_feature_dim, old_param.shape[1], new_param.shape[1])),
                    )
                elif param_name == "critic.0.weight":
                    base_feature_dim = self.stage.num_critic_observations - getattr(
                        self.stage,
                        "num_goal_obs",
                        0,
                    )
                    copy_slices = (
                        slice(0, min(old_param.shape[0], new_param.shape[0])),
                        slice(0, min(base_feature_dim, old_param.shape[1], new_param.shape[1])),
                    )
                new_param[copy_slices] = old_param[copy_slices]
            partial_keys.append(
                f"{param_name} {list(old_param.shape)} -> {list(new_param.shape)}"
            )

        model.load_state_dict(current_state)

        if self.logger is not None:
            self.logger.info(
                f"Partial load model {model_file_path}: "
                f"{len(loaded_keys)} exact, {len(partial_keys)} partial, {len(skipped_keys)} skipped"
            )
            for partial_info in partial_keys:
                self.logger.info(f"  Partial: {partial_info}")
