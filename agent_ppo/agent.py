#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
import os

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

import torch.optim as optim

from kaiwudrl.interface.agent import BaseAgent
from agent_ppo.feature.definition import ActData
from agent_ppo.conf.conf import Config, _load_toml
from agent_ppo.model.actor_critic import ActorCritic
from agent_ppo.algorithm.algorithm_ppo import AlgorithmPPO
from tools.train_env_conf_validate import check_usr_conf


def _obs_height_grid(obs, scan_start: int = 45, scan_size: int = 256):
    if obs is None or not hasattr(obs, "shape") or obs.shape[-1] < scan_start + scan_size:
        return None
    side = int(scan_size ** 0.5)
    if side * side != scan_size:
        return None
    return obs[:, scan_start:scan_start + scan_size].view(obs.shape[0], side, side)


def _classify_pre_maze_terrain(obs, rl_nav_conf):
    grid = _obs_height_grid(
        obs,
        scan_start=int(rl_nav_conf.get("scan_start", 45)),
        scan_size=int(rl_nav_conf.get("scan_size", 256)),
    )
    if grid is None:
        return None

    row_start = max(int(rl_nav_conf.get("terrain_row_start", 3)), 0)
    row_end = min(int(rl_nav_conf.get("terrain_row_end", 13)), grid.shape[1])
    front_cols = min(int(rl_nav_conf.get("terrain_front_cols", 8)), grid.shape[2])
    if row_end <= row_start or front_cols <= 1:
        return None

    sector = grid[:, row_start:row_end, :front_cols]
    if sector.shape[1] == 0 or sector.shape[2] <= 1:
        return None

    lateral_std = sector.std(dim=1, unbiased=False).mean(dim=1)
    dx = sector[:, :, 1:] - sector[:, :, :-1]
    abs_dx = dx.abs()
    if abs_dx.numel() == 0:
        return None

    q = float(rl_nav_conf.get("terrain_step_quantile", 0.85))
    q = min(max(q, 0.0), 1.0)
    step_strength = torch.quantile(abs_dx.flatten(1), q, dim=1)
    sign_consistency = dx.mean(dim=(1, 2)).abs() / (abs_dx.mean(dim=(1, 2)) + 1e-6)
    if dx.shape[2] > 1:
        second_diff = (dx[:, :, 1:] - dx[:, :, :-1]).abs().mean(dim=(1, 2))
    else:
        second_diff = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)

    is_uniform = lateral_std < float(rl_nav_conf.get("terrain_lateral_std_threshold", 0.18))
    not_wall = sector.amin(dim=(1, 2)) > float(rl_nav_conf.get("terrain_wall_height_threshold", -1.05))
    terrain_like = is_uniform & not_wall & (
        step_strength > float(rl_nav_conf.get("terrain_slope_delta_threshold", 0.035))
    )
    stair_like = terrain_like & (
        (step_strength > float(rl_nav_conf.get("terrain_stair_delta_threshold", 0.10)))
        | (second_diff > float(rl_nav_conf.get("terrain_stair_second_diff_threshold", 0.055)))
    )
    slope_like = terrain_like & ~stair_like & (
        sign_consistency > float(rl_nav_conf.get("terrain_slope_sign_consistency_threshold", 0.55))
    )

    terrain_id = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
    terrain_id = torch.where(slope_like, torch.ones_like(terrain_id), terrain_id)
    terrain_id = torch.where(stair_like, torch.full_like(terrain_id, 2), terrain_id)
    return terrain_id


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device="cuda", logger=None, monitor=None):
        self.cur_model_name = "ActorCritic"
        self.device = device
        self.logger = logger
        self.monitor = monitor

        usr_conf, usr_conf_file, is_eval, stage = Config.load_conf(self.logger)
        self.is_eval = is_eval
        valid, message = check_usr_conf(usr_conf, is_eval, self.logger)
        if not valid:
            self.logger.error(f"check_usr_conf is {valid}, message is {message}, please check {usr_conf_file}")
            raise Exception(f"check_usr_conf is {valid}, message is {message}, please check {usr_conf_file}")

        self.stage = stage
        env_conf = usr_conf["env"]
        self.num_envs = env_conf["num_envs"]

        # Model architecture dims come from StageConfig (architecture constants,
        # not user-tunable business params). Do NOT read them from TOML.
        # 模型架构维度来自 StageConfig（架构常量，非业务可调参数），不从 TOML 读。
        self.num_actions = stage.num_actions
        self.num_critic_obs = stage.num_critic_observations

        num_proprio = stage.num_proprio_obs
        num_scan = stage.num_scan
        num_goal_obs = getattr(stage, "num_goal_obs", 0)

        # policy obs = proprio + scan + goal
        # 策略观测 = 本体感知 + 扫描 + 目标
        self.num_obs = num_proprio + num_scan + num_goal_obs

        self._init_flat(num_proprio, num_scan, num_goal_obs, stage)

        train_stage_conf = {}
        if is_eval and stage.task_type == "track":
            train_stage_conf = self._load_train_stage_conf(stage)

        # Pure-RL navigation: no local planner is used.  In evaluation, platform
        # command ranges may be unrelated to this training objective, so keep the
        # policy command observation aligned with the forward-walking anchor.
        self.eval_command_override = None
        self.eval_phase_command_enabled = False
        self.eval_pre_maze_command = None
        self.eval_slope_command = None
        self.eval_stairs_command = None
        self.eval_maze_command = None
        self.eval_phase_maze_goal_dist_gate = 14.0
        self.eval_rl_nav_conf = {}
        if is_eval and stage.task_type == "track":
            rl_nav_conf = train_stage_conf.get("rl_navigation", {}).copy()
            rl_nav_conf.update(usr_conf.get("rl_navigation", {}))
            self.eval_rl_nav_conf = rl_nav_conf.copy()
            self.eval_phase_command_enabled = bool(rl_nav_conf.get("phase_command_enabled", False))
            self.eval_phase_maze_goal_dist_gate = float(
                rl_nav_conf.get("phase_maze_goal_dist_gate", 14.0)
            )

            def _eval_phase_command(command_key, range_key, default_range):
                """Build one fixed eval command for a track phase.

                Args:
                    command_key: Optional explicit 3D command in TOML, such as
                        ``eval_stairs_command = [0.58, 0, 0]``.
                    range_key: Training speed range to fall back to when the
                        explicit eval command is absent.
                    default_range: Last fallback range if neither key exists.

                Evaluation is deterministic, so a fixed command is preferable to
                sampling from the train range.  It lets us tune time_score versus
                energy_score per phase without retraining the observation shape.
                """
                command = rl_nav_conf.get(command_key)
                if isinstance(command, (list, tuple)) and len(command) == 3:
                    # Preferred path: use the exact eval command from TOML.
                    values = [float(command[0]), float(command[1]), float(command[2])]
                    return torch.tensor(values, device=self.device, dtype=torch.float32)

                speed_range = rl_nav_conf.get(range_key, default_range)
                if isinstance(speed_range, (list, tuple)) and len(speed_range) == 2:
                    # Fallback path: use the midpoint of the training speed range.
                    vx = 0.5 * (float(speed_range[0]) + float(speed_range[1]))
                    return torch.tensor([vx, 0.0, 0.0], device=self.device, dtype=torch.float32)

                # Invalid or missing config: leave this phase command disabled.
                return None

            pre_range = rl_nav_conf.get("pre_maze_lin_vel_x", [0.75, 1.0])
            # Each phase can have a separate deterministic eval command.  This
            # is useful because stairs usually need lower speed for energy and
            # stability, while slope/pre-maze can afford higher speed.
            self.eval_pre_maze_command = _eval_phase_command(
                "eval_pre_maze_command", "pre_maze_lin_vel_x", pre_range
            )
            self.eval_slope_command = _eval_phase_command(
                "eval_slope_command", "slope_lin_vel_x", pre_range
            )
            self.eval_stairs_command = _eval_phase_command(
                "eval_stairs_command", "stairs_lin_vel_x", pre_range
            )
            self.eval_maze_command = _eval_phase_command(
                "eval_maze_command", "maze_lin_vel_x", [0.45, 0.65]
            )
            if bool(rl_nav_conf.get("eval_command_override", True)):
                cmd = rl_nav_conf.get("eval_command", [0.55, 0.0, 0.0])
                if len(cmd) == 3:
                    self.eval_command_override = torch.tensor(
                        cmd, device=self.device, dtype=torch.float32
                    )
                    self.logger.info(
                        "[RLNavigation] Eval policy command obs override enabled: "
                        f"{cmd}"
                    )

        self.num_steps_per_env = stage.num_steps_per_env
        self.save_interval = stage.model_save_interval

        # Initialize storage
        # 初始化存储
        self.algorithm.init_storage(
            self.num_envs,
            self.num_steps_per_env,
            actor_obs_shape=(self.num_obs,),
            critic_obs_shape=(self.num_critic_obs,),
            action_shape=(self.num_actions,),
            device=self.device,
        )

        super().__init__(agent_type, device, logger, monitor)

    def _load_train_stage_conf(self, stage):
        train_conf_file = f"agent_ppo/conf/train_env_conf_{stage.task_type}_{stage.name}.toml"
        if not os.path.exists(train_conf_file):
            return {}
        try:
            return _load_toml(train_conf_file)
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning(
                    f"[RLNavigation] Failed to load train stage config "
                    f"from {train_conf_file}: {exc}"
                )
            return {}

    def _apply_eval_command_to_obs(self, obs):
        """Patch command observations during evaluation.

        The platform may supply broad/random velocity commands during eval, but
        this Track policy was trained to use command slots as a gait anchor and
        goal/scan features for navigation.  Replacing obs[:, 6:9] keeps eval
        consistent with training and avoids wasting energy chasing irrelevant
        lateral/yaw commands.
        """
        if obs is None or obs.shape[-1] < 9:
            return obs
        if (
            self.eval_phase_command_enabled
            and self.eval_pre_maze_command is not None
            and self.eval_maze_command is not None
            and obs.shape[-1] >= 304
        ):
            nav_obs = obs.clone()
            # Goal distance is appended by PolicyObservationProcess as the last
            # goal feature, normalized by /20.  Undo that normalization to detect
            # the final maze phase.
            goal_dist = torch.clamp(nav_obs[:, 303], 0.0, 1.0) * 20.0
            maze_phase = goal_dist < self.eval_phase_maze_goal_dist_gate

            # Start from pre-maze speed, then override by terrain classifier or
            # maze phase below.
            command = self.eval_pre_maze_command.to(device=obs.device, dtype=obs.dtype).expand(obs.shape[0], -1)
            if bool(self.eval_rl_nav_conf.get("terrain_phase_speed_enabled", False)):
                # Height-scan classifier distinguishes slope/stairs before the
                # maze so eval can slow down only where it matters.
                terrain_id = _classify_pre_maze_terrain(nav_obs, self.eval_rl_nav_conf)
                if terrain_id is not None:
                    if self.eval_slope_command is not None:
                        slope_command = self.eval_slope_command.to(device=obs.device, dtype=obs.dtype).expand(obs.shape[0], -1)
                        command = torch.where((terrain_id == 1).unsqueeze(1), slope_command, command)
                    if self.eval_stairs_command is not None:
                        stairs_command = self.eval_stairs_command.to(device=obs.device, dtype=obs.dtype).expand(obs.shape[0], -1)
                        command = torch.where((terrain_id == 2).unsqueeze(1), stairs_command, command)
            maze_command = self.eval_maze_command.to(device=obs.device, dtype=obs.dtype).expand(obs.shape[0], -1)
            # Maze phase has final priority because wall avoidance and turning
            # are usually more expensive than straight pre-maze running.
            command = torch.where(maze_phase.unsqueeze(1), maze_command, command)

            # In the standard Isaac Lab observation layout, command slots are
            # obs[:, 6:9] = [lin_vel_x, lin_vel_y, yaw_rate].
            nav_obs[:, 6:9] = command
            return nav_obs
        if self.eval_command_override is None:
            return obs
        nav_obs = obs.clone()
        # Coarse fallback: one fixed command for the whole track.
        nav_obs[:, 6:9] = self.eval_command_override.to(
            device=obs.device, dtype=obs.dtype
        )
        return nav_obs

    def _init_flat(self, num_proprio, num_scan, num_goal_obs, stage):
        """
        Initialize single-model (flat) architecture.
        初始化单模型（扁平）架构。
        """
        self.model = ActorCritic(
            num_obs=self.num_obs,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.num_actions,
            actor_hidden_dims=stage.actor_hidden_dims,
            critic_hidden_dims=stage.critic_hidden_dims,
            activation=stage.activation,
            init_noise_std=getattr(stage, "init_noise_std", 1.0),
        ).to(self.device)

        self.logger.info(f"Actor MLP: {self.model.actor}")
        self.logger.info(f"Critic MLP: {self.model.critic}")

        params = [{"params": self.model.parameters(), "name": "actor_critic"}]
        self.optimizer = optim.Adam(params, lr=stage.lr)

        self.algorithm = AlgorithmPPO(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            logger=self.logger,
            monitor=self.monitor,
            learning_rate=stage.lr,
            clip_param=getattr(stage, "clip_param", 0.2),
            entropy_coef=getattr(stage, "entropy_coef", 0.01),
            desired_kl=getattr(stage, "desired_kl", 0.01),
            num_mini_batches=stage.num_mini_batches,
            num_learning_epochs=stage.num_learning_epochs,
        )

    def exploit(self, list_obs_data):
        """
        Exploit learned policy for action selection in evaluation mode.
        在评估模式下利用已学习的策略进行动作选择。
        """
        (obs) = list_obs_data
        with torch.no_grad():
            obs = self._apply_eval_command_to_obs(obs)
            actions = self.algorithm.actor_critic.act_inference(obs)
            return [ActData(action=actions)]

    def learn(self, list_sample_data=None):
        """
        Trigger learning process using sample data.
        使用样本数据触发学习过程。

        Note: AlgorithmPPO.learn() doesn't take batch_data as argument anymore.
        It reads from its internal storage that was filled by workflow's run_episodes_.
        注：AlgorithmPPO.learn() 不再接受 batch_data 参数，
        而是直接读取 workflow 的 run_episodes_ 填充的内部存储。
        """
        return self.algorithm.learn()

    def predict(self, list_obs_data):
        """
        Generate predictions with actor-critic network.
        使用 actor-critic 网络生成预测。
        """
        (obs, critic_obs) = list_obs_data

        with torch.no_grad():
            if self.is_eval:
                obs = self._apply_eval_command_to_obs(obs)
            hidden_states = None
            if getattr(self.algorithm.actor_critic, "is_recurrent", False):
                current_hidden = self.algorithm.actor_critic.get_hidden_states()
                if current_hidden is None or current_hidden[0].shape[1] != obs.shape[0]:
                    self.algorithm.actor_critic._init_hidden_states(obs.shape[0], obs.device, obs.dtype)
                    current_hidden = self.algorithm.actor_critic.get_hidden_states()
                hidden_states = tuple(state.detach().clone() for state in current_hidden)

            actions = self.algorithm.actor_critic.act(obs)
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
                obs.detach(),
                critic_obs.detach(),
                hidden_states,
            )

    def save_model(self, path=None, id="1"):
        """
        Save model checkpoint.
        保存模型 checkpoint。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """
        Load model checkpoint.
        加载模型 checkpoint。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
            return

        pretrained = torch.load(model_file_path, map_location=self.device)
        current_state = self.model.state_dict()

        has_mismatch = False
        for key in pretrained:
            if key in current_state and pretrained[key].shape != current_state[key].shape:
                has_mismatch = True
                break

        if not has_mismatch:
            self.model.load_state_dict(pretrained)
            self.logger.info(f"load model {model_file_path} successfully (exact match)")
        else:
            self._load_model_partial(self.model, pretrained, model_file_path)

        self._enforce_action_std_bounds()
        self.cur_model_name = model_file_path

    def _enforce_action_std_bounds(self):
        min_std_cfg = getattr(self.stage, "min_normalized_std", None)
        max_std_cfg = getattr(self.stage, "max_normalized_std", None)
        if min_std_cfg is None and max_std_cfg is None:
            return

        def _bound_std(std_tensor):
            posinf_value = 1.0e6
            if max_std_cfg is not None:
                max_std = torch.tensor(max_std_cfg, device=self.device, dtype=std_tensor.dtype)
                if max_std.shape == std_tensor.data.shape:
                    posinf_value = float(torch.max(max_std).item())
            bounded = torch.nan_to_num(
                std_tensor.data,
                nan=1.0,
                posinf=posinf_value,
                neginf=0.0,
            )
            if min_std_cfg is not None:
                min_std = torch.tensor(min_std_cfg, device=self.device, dtype=std_tensor.dtype)
                if min_std.shape == bounded.shape:
                    bounded = torch.maximum(bounded, min_std)
            if max_std_cfg is not None:
                max_std = torch.tensor(max_std_cfg, device=self.device, dtype=std_tensor.dtype)
                if max_std.shape == bounded.shape:
                    bounded = torch.minimum(bounded, max_std)
            std_tensor.data.copy_(bounded)

        with torch.no_grad():
            if hasattr(self.model, "std"):
                _bound_std(self.model.std)
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
                self.logger.info(
                    f"[PPO] log action std bounds enforced: min={min_std_cfg}, max={max_std_cfg}"
                )

    def _load_model_partial(self, model, pretrained, model_file_path):
        """
        Partial checkpoint loading for cross-stage transfer.
        部分加载 checkpoint，用于跨阶段迁移。
        """
        current_state = model.state_dict()
        loaded_keys = []
        partial_keys = []
        skipped_keys = []

        for key in current_state:
            if key not in pretrained:
                skipped_keys.append(key)
                continue

            if getattr(model, "is_recurrent", False) and key in {"actor.0.weight", "actor.0.bias"}:
                skipped_keys.append(f"{key} (recurrent front-end)")
                continue

            old_param = pretrained[key]
            new_param = current_state[key]

            if old_param.shape == new_param.shape:
                new_param.copy_(old_param)
                loaded_keys.append(key)
            else:
                with torch.no_grad():
                    new_param.zero_()
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_param.shape, new_param.shape))
                    if key == "actor.0.weight":
                        base_cols = self.stage.num_proprio_obs + self.stage.num_scan
                        slices = (
                            slice(0, min(old_param.shape[0], new_param.shape[0])),
                            slice(0, min(base_cols, old_param.shape[1], new_param.shape[1])),
                        )
                    elif key == "critic.0.weight":
                        base_cols = self.stage.num_critic_observations - getattr(self.stage, "num_goal_obs", 0)
                        slices = (
                            slice(0, min(old_param.shape[0], new_param.shape[0])),
                            slice(0, min(base_cols, old_param.shape[1], new_param.shape[1])),
                        )
                    new_param[slices] = old_param[slices]
                partial_keys.append(f"{key} {list(old_param.shape)}→{list(new_param.shape)}")

        model.load_state_dict(current_state)

        self.logger.info(
            f"Partial load model {model_file_path}: "
            f"{len(loaded_keys)} exact, {len(partial_keys)} partial, {len(skipped_keys)} skipped"
        )
        for info in partial_keys:
            self.logger.info(f"  Partial: {info}")
