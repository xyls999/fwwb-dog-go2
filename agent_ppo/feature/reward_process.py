# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
RewardProcess — PPO locomotion and pure-RL maze navigation rewards.

The locomotion terms keep the pretrained gait stable.  The navigation terms
shape target-facing progress, completion, wall avoidance, and a very small
goal-directed exploration signal without using a hand-written planner.
"""

import torch

from tools.base_env.base_reward import RewardProcessBase


class RewardProcess(RewardProcessBase):

    def _tracking_command(self, command_name: str = "base_velocity"):
        return self.env.command_manager.get_command(command_name)

    @staticmethod
    def _quat_to_roll_pitch(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract roll and pitch from WXYZ quaternions, matching BaseScorer."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)
        return roll, pitch

    # -----------------------------------------------------------------------
    # Locomotion quality rewards
    # 运动质量奖励
    # -----------------------------------------------------------------------

    def _reward_track_lin_vel_xy(self, std: float = 0.25, command_name: str = "base_velocity"):
        asset = self._get_robot_asset()
        cmd = self._tracking_command(command_name)
        error = cmd[:, :2] - asset.data.root_lin_vel_b[:, :2]
        return torch.exp(-torch.sum(torch.square(error), dim=1) / max(std * std, 1e-6))

    def _reward_command_speed_advantage(
        self,
        command_name: str = "base_velocity",
        deadband: float = 0.03,
        surplus_scale: float = 0.35,
        lag_scale: float = 0.35,
        max_surplus: float = 0.60,
        max_lag: float = 0.60,
        lag_penalty_scale: float = 1.0,
        min_command: float = 0.10,
    ):
        """Signed forward-speed reward around the published command.

        If actual vx is below commanded vx, this returns a negative penalty.
        If actual vx is above commanded vx, this returns a positive reward that
        grows with surplus speed, with a cap to avoid overwhelming posture.
        """
        asset = self._get_robot_asset()
        cmd_vx = self._tracking_command(command_name)[:, 0]
        actual_vx = asset.data.root_lin_vel_b[:, 0]

        surplus = actual_vx - cmd_vx
        faster = torch.clamp(surplus - deadband, min=0.0, max=max_surplus) / max(surplus_scale, 1e-6)
        slower = torch.clamp(-surplus - deadband, min=0.0, max=max_lag) / max(lag_scale, 1e-6)
        active = cmd_vx > min_command
        return active.float() * (faster - lag_penalty_scale * slower)

    def _reward_track_ang_vel_z(self, std: float = 0.25, command_name: str = "base_velocity"):
        asset = self._get_robot_asset()
        cmd = self._tracking_command(command_name)
        error = cmd[:, 2] - asset.data.root_ang_vel_b[:, 2]
        return torch.exp(-torch.square(error / max(std, 1e-6)))

    def _reward_feet_air_time(self, command_name: str = "base_velocity", threshold: float = 0.5):
        """Reward long steps (feet air time above threshold when moving).

        奖励长步幅（移动时脚部滞空时间超过阈值）。
        Ref: Rudin et al., "Learning to Walk in Minutes", RSS 2022 (legged_gym).
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        if contact_sensor.cfg.track_air_time is False:
            raise RuntimeError("Activate ContactSensor's track_air_time!")
        first_contact = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] == 0.0
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
        is_moving = torch.norm(self._tracking_command(command_name)[:, :2], dim=1) > 0.1
        return reward * is_moving.float()

    def _reward_feet_clearance(
        self,
        command_name: str = "base_velocity",
        target_height: float = 0.08,
        std: float = 0.05,
        terrain_height_scale: float = 0.6,
        max_terrain_extra_height: float = 0.08,
        speed_height_scale: float = 0.01,
        body_y_start: int = 5,
        body_y_end: int = 11,
        near_x_start: int = 2,
        near_x_end: int = 10,
        delta_quantile: float = 0.85,
    ):
        """Reward terrain-aware swing-foot clearance to reduce stair-edge tripping.

        Active only for moving commands and swing feet. The Gaussian target keeps
        the reward bounded: it encourages enough clearance to step over edges,
        but does not reward unnecessarily high, energy-wasting leg lifts.

        The dynamic target is based on local adjacent height-scan deltas, not a
        full-window max-min range, so several stair levels in the scan window are
        less likely to be accumulated into one exaggerated step height.
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]

        contact_forces = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
        )
        swing = contact_forces <= 1.0
        foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - asset.data.root_pos_w[:, 2].unsqueeze(1)
        command = self._tracking_command(command_name)
        command_speed = torch.norm(command[:, :2], dim=1)

        terrain_extra = torch.zeros(self.env.num_envs, device=self.env.device)
        height_scanner = self.env.scene.sensors.get("height_scanner")
        if height_scanner is not None:
            scan = height_scanner.data.pos_w[:, 2:3] - height_scanner.data.ray_hits_w[..., 2]
            grid = scan.view(self.env.num_envs, 16, 16)
            forward_window = grid[:, body_y_start:body_y_end, near_x_start:near_x_end]
            if forward_window.shape[-1] > 1 and forward_window.shape[1] > 0:
                step_deltas = torch.abs(forward_window[:, :, 1:] - forward_window[:, :, :-1]).flatten(1)
                local_step = torch.quantile(step_deltas, delta_quantile, dim=1)
                terrain_extra = torch.clamp(
                    terrain_height_scale * local_step,
                    0.0,
                    max_terrain_extra_height,
                )

        speed_extra = speed_height_scale * torch.clamp(command_speed, 0.0, 1.0)
        dynamic_target_height = target_height + terrain_extra + speed_extra
        height_error = (foot_height - dynamic_target_height.unsqueeze(1)) / max(std, 1e-6)
        clearance_reward = torch.exp(-torch.square(height_error))
        is_moving = command_speed > 0.1
        return torch.sum(clearance_reward * swing.float(), dim=1) * is_moving.float() / max(len(asset_cfg.body_ids), 1)

    def _reward_feet_swing_forward(
        self,
        command_name: str = "base_velocity",
        target_forward: float = 0.10,
        std: float = 0.08,
        min_command: float = 0.10,
    ):
        """Reward swing feet moving forward enough to follow the body on stairs.

        This complements feet_clearance: clearance helps the foot avoid stair
        edges vertically, while this term encourages the swing foot—especially
        the rear feet—to reach forward instead of taking a short high step that
        still lands on the stair edge.
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]

        contact_forces = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
        )
        swing = contact_forces <= 1.0
        foot_forward = asset.data.body_pos_w[:, asset_cfg.body_ids, 0] - asset.data.root_pos_w[:, 0].unsqueeze(1)
        shortfall = torch.clamp(target_forward - foot_forward, min=0.0)
        forward_reward = torch.exp(-torch.square(shortfall / max(std, 1e-6)))

        command = self._tracking_command(command_name)
        has_forward_command = command[:, 0] > min_command
        return torch.sum(forward_reward * swing.float(), dim=1) * has_forward_command.float() / max(len(asset_cfg.body_ids), 1)

    def _reward_running_speed_band(
        self,
        command_name: str = "base_velocity",
        min_speed: float = 1.25,
        target_speed: float = 2.05,
        max_speed: float = 2.55,
        posture_limit: float = 0.24,
    ):
        """Reward actual high forward speed while gating out wild body tilt."""
        asset = self._get_robot_asset()
        cmd_vx = self._tracking_command(command_name)[:, 0]
        vx = asset.data.root_lin_vel_b[:, 0]

        speed_score = torch.clamp((vx - min_speed) / max(target_speed - min_speed, 1e-6), 0.0, 1.0)
        overspeed = torch.clamp((vx - max_speed) / max(max_speed - target_speed, 1e-6), 0.0, 1.0)

        roll, pitch = self._quat_to_roll_pitch(asset.data.root_quat_w)
        pose_deviation = torch.abs(roll) + torch.abs(pitch)
        posture_gate = torch.exp(-torch.square(pose_deviation / max(posture_limit, 1e-6)))
        active = cmd_vx > min_speed
        return active.float() * speed_score * posture_gate * (1.0 - 0.5 * overspeed)

    def _reward_running_stride_span(
        self,
        command_name: str = "base_velocity",
        target_span: float = 0.62,
        std: float = 0.16,
        min_command: float = 1.25,
        min_body_speed: float = 1.15,
    ):
        """Reward long fore-aft foot span in the robot body-forward direction."""
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]

        contact_forces = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
        )
        swing = contact_forces <= 1.0

        quat = asset.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        body_x_w = torch.stack(
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + w * z),
                2.0 * (x * z - w * y),
            ),
            dim=1,
        )
        rel_foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :3].unsqueeze(1)
        foot_forward = torch.sum(rel_foot_pos * body_x_w.unsqueeze(1), dim=2)
        span = foot_forward.max(dim=1).values - foot_forward.min(dim=1).values
        shortfall = torch.clamp(target_span - span, min=0.0)
        span_score = torch.exp(-torch.square(shortfall / max(std, 1e-6)))

        command = self._tracking_command(command_name)
        active = (command[:, 0] > min_command) & (asset.data.root_lin_vel_b[:, 0] > min_body_speed)
        has_swing = torch.any(swing, dim=1)
        return active.float() * has_swing.float() * span_score

    def _reward_running_contact_pattern(
        self,
        command_name: str = "base_velocity",
        target_contacts: float = 2.0,
        contact_std: float = 0.85,
        min_command: float = 1.25,
        min_body_speed: float = 1.15,
    ):
        """Encourage a dynamic two-foot support pattern instead of fast walking."""
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self._get_robot_asset()
        contact_forces = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
        )
        contact_count = torch.sum((contact_forces > 1.0).float(), dim=1)
        contact_score = torch.exp(-torch.square((contact_count - target_contacts) / max(contact_std, 1e-6)))
        active = (self._tracking_command(command_name)[:, 0] > min_command) & (
            asset.data.root_lin_vel_b[:, 0] > min_body_speed
        )
        return active.float() * contact_score

    def _reward_running_air_time_band(
        self,
        command_name: str = "base_velocity",
        min_air_time: float = 0.12,
        target_air_time: float = 0.24,
        max_air_time: float = 0.42,
        min_command: float = 1.25,
    ):
        """Reward bounded swing air time so the gait can leave fast walking."""
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self._get_robot_asset()
        current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
        best_air_time = current_air_time.max(dim=1).values
        air_score = torch.clamp(
            (best_air_time - min_air_time) / max(target_air_time - min_air_time, 1e-6),
            0.0,
            1.0,
        )
        too_long = torch.clamp(
            (best_air_time - max_air_time) / max(max_air_time - target_air_time, 1e-6),
            0.0,
            1.0,
        )
        active = (self._tracking_command(command_name)[:, 0] > min_command) & (asset.data.root_lin_vel_b[:, 0] > 1.0)
        return active.float() * air_score * (1.0 - too_long)

    def _reward_running_speed_efficiency(
        self,
        command_name: str = "base_velocity",
        target_speed: float = 2.05,
        power_scale: float = 85.0,
        min_command: float = 1.25,
    ):
        """Reward getting high forward speed without completely ignoring power."""
        asset = self._get_robot_asset()
        vx_score = torch.clamp(asset.data.root_lin_vel_b[:, 0] / max(target_speed, 1e-6), 0.0, 1.0)
        power = torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)
        efficiency_score = torch.exp(-power / max(power_scale, 1e-6))
        active = self._tracking_command(command_name)[:, 0] > min_command
        return active.float() * vx_score * efficiency_score

    def _reward_feet_slide(self):
        """Penalize feet sliding on the ground (velocity while in contact).

        惩罚脚部在地面上的滑动（接触时的速度）。
        使用 net_forces_w_history 的 3D 力范数 + 多帧取 max 判定接触。
        Ref: Miki et al., Science Robotics 2022.
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        asset_cfg = self._get_foot_asset_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        asset = self.env.scene[asset_cfg.name]
        contacts = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        )
        body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

    def _reward_joint_position_penalty(
        self,
        stand_still_scale: float = 5.0,
        velocity_threshold: float = 0.1,
        cmd_threshold: float = 0.1,
        ang_cmd_threshold: float = 0.2,
    ):
        """Penalize joint position deviation from default pose.

        惩罚关节位置偏离默认姿态；静止时惩罚倍增（鼓励站立保持默认姿势）。
        Ref: Kumar et al., "RMA: Rapid Motor Adaptation", RSS 2021.

        Bug fix: original code used `cmd > 0.0` (exact-zero check).
        Since velocity commands are sampled from continuous distributions, the
        probability of all three components being EXACTLY 0.0 simultaneously is
        essentially zero — stand_still_scale never fired in practice.
        Fixed to use `cmd > cmd_threshold` (default 0.1 m/s equivalent) so the
        scale activates whenever the robot is genuinely commanded to stand still.

        原始实现使用 `cmd > 0.0`（精确零判断）。由于速度指令来自连续分布，
        三个分量同时精确等于 0 的概率几乎为零，stand_still_scale 实际上从不生效。
        修复为 `cmd > cmd_threshold`（默认 0.1），让真正的零速指令时才放大惩罚。

        Args:
            stand_still_scale: Penalty multiplier when standing still (cmd ≈ 0).
                               静止（cmd ≈ 0）时的惩罚倍数。
            velocity_threshold: Body velocity threshold to confirm robot is not moving (m/s).
                                机体速度阈值，低于此值判定为静止 (m/s)。
            cmd_threshold: Command norm threshold below which robot is "standing still" (m/s).
                           速度指令模长阈值，低于此值视为零速指令 (m/s)。
            ang_cmd_threshold: Yaw-rate command threshold below which robot is treated as not turning.
                               偏航角速度指令阈值，低于此值视为未被要求转向。
        """
        asset = self._get_robot_asset()
        cmd = self._tracking_command("base_velocity")
        cmd_xy = torch.linalg.norm(cmd[:, :2], dim=1)
        cmd_yaw = torch.abs(cmd[:, 2])
        body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
        deviation = torch.linalg.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1)
        is_moving = torch.logical_or(
            torch.logical_or(cmd_xy > cmd_threshold, cmd_yaw > ang_cmd_threshold),
            body_vel > velocity_threshold,
        )
        return torch.where(
            is_moving,
            deviation,
            stand_still_scale * deviation,
        )

    def _reward_stand_still_motion(
        self,
        command_name: str = "base_velocity",
        lin_cmd_threshold: float = 0.15,
        ang_cmd_threshold: float = 0.2,
        vertical_vel_scale: float = 0.5,
        ang_vel_scale: float = 0.5,
        joint_vel_scale: float = 0.1,
    ):
        """Penalize body oscillation and leg fidgeting under near-zero commands.

        连续命令采样几乎不会精确落在 0 速度，因此“站立不动”往往训练不足。
        本奖励在近零线速度/角速度命令带内激活，直接惩罚机身上下晃动、
        pitch/roll 摇摆以及小腿频繁抬放造成的关节速度。

        Args:
            lin_cmd_threshold: Near-zero threshold for XY linear command norm (m/s).
            ang_cmd_threshold: Near-zero threshold for yaw command magnitude (rad/s).
            vertical_vel_scale: Weight for vertical body velocity in the penalty.
            ang_vel_scale: Weight for pitch/roll angular velocity in the penalty.
            joint_vel_scale: Weight for mean absolute joint velocity in the penalty.
        """
        asset = self._get_robot_asset()
        cmd = self._tracking_command(command_name)

        near_zero_cmd = (
            torch.linalg.norm(cmd[:, :2], dim=1) < lin_cmd_threshold
        ) & (
            torch.abs(cmd[:, 2]) < ang_cmd_threshold
        )

        base_lin_vel = asset.data.root_lin_vel_b
        base_ang_vel = asset.data.root_ang_vel_b
        mean_abs_joint_vel = torch.mean(torch.abs(asset.data.joint_vel), dim=1)

        motion_penalty = (
            torch.linalg.norm(base_lin_vel[:, :2], dim=1)
            + vertical_vel_scale * torch.abs(base_lin_vel[:, 2])
            + ang_vel_scale * torch.linalg.norm(base_ang_vel[:, :2], dim=1)
            + joint_vel_scale * mean_abs_joint_vel
        )
        return near_zero_cmd.float() * motion_penalty

    def _reward_commanded_still_penalty(
        self,
        command_name: str = "base_velocity",
        cmd_threshold: float = 0.20,
        still_speed_threshold: float = 0.08,
    ):
        """Penalize staying nearly still when an XY velocity command is present."""
        asset = self._get_robot_asset()
        cmd = self._tracking_command(command_name)
        cmd_speed = torch.linalg.norm(cmd[:, :2], dim=1)
        body_speed = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

        commanded_to_move = cmd_speed > cmd_threshold
        stillness = torch.clamp(
            (still_speed_threshold - body_speed) / max(still_speed_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )
        return commanded_to_move.float() * stillness

    def _reward_score_guidance(
        self,
        command_name: str = "base_velocity",
        min_command: float = 0.15,
        tracking_std: float = 0.35,
        posture_std: float = 0.25,
        power_scale: float = 35.0,
        posture_weight: float = 0.6,
    ):
        """Small bounded bonus aligned with time, posture, and energy scores.

        The tracking gate prevents the policy from earning the posture/energy bonus
        by standing still when it has a movement command.
        """
        asset = self._get_robot_asset()
        cmd = self._tracking_command(command_name)

        cmd_xy = cmd[:, :2]
        actual_xy = asset.data.root_lin_vel_b[:, :2]
        cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
        moving_cmd = cmd_speed > min_command

        vel_error = torch.sum(torch.square(cmd_xy - actual_xy), dim=1)
        tracking_score = torch.exp(-vel_error / max(tracking_std * tracking_std, 1e-6))

        roll, pitch = self._quat_to_roll_pitch(asset.data.root_quat_w)
        pose_deviation = torch.abs(roll) + torch.abs(pitch)
        pose_deviation = torch.nan_to_num(pose_deviation, nan=0.0, posinf=0.0, neginf=0.0)
        posture_score = torch.exp(-5.0 * pose_deviation)

        power = torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)
        energy_score = torch.exp(-power / max(power_scale, 1e-6))

        posture_weight = min(max(posture_weight, 0.0), 1.0)
        score_hint = posture_weight * posture_score + (1.0 - posture_weight) * energy_score
        return moving_cmd.float() * tracking_score * score_hint

    def _reward_feet_stumble(self):
        """Penalize feet hitting vertical surfaces (stair edges, walls).

        惩罚脚撞到垂直面（台阶边缘、墙壁）。水平力 > 5× 垂直力时触发。
        阈值 5× 对齐 legged_gym 原版。
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
        forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        return torch.any(forces_xy > 5 * forces_z, dim=1).float()

    def _reward_termination(self):
        """Penalize real failures (terminated AND NOT timed-out).

        惩罚真正的失败（被终止 且 非超时截断），对应 legged_gym `reset_buf * ~time_out_buf` 逻辑。
        防止策略学会"倒地不起"以规避其他惩罚。
        """
        term_mgr = self.env.termination_manager
        failure = term_mgr.terminated & ~term_mgr.time_outs
        try:
            if "goal_reached" in term_mgr.active_terms:
                failure = failure & ~term_mgr.get_term("goal_reached")
        except Exception:
            pass
        return failure.float()

    def _reward_non_completion_timeout(self):
        """Penalize timeout episodes that did not reach the Track goal."""
        term_mgr = self.env.termination_manager
        timeout = term_mgr.time_outs
        try:
            if "goal_reached" in term_mgr.active_terms:
                timeout = timeout & ~term_mgr.get_term("goal_reached")
        except Exception:
            pass
        return timeout.float()

    def _reward_action_smoothness(self):
        """2nd-order action smoothness penalty (squared action acceleration).

        动作二阶平滑惩罚（动作加速度的平方和）。
        a_t - 2*a_{t-1} + a_{t-2} 的平方和，比一阶 action_rate 更能抑制抖动。
        在 env 上缓存 prev_prev_action 以跨调用维持状态。
        """
        curr = self.env.action_manager.action
        prev = self.env.action_manager.prev_action
        if not hasattr(self.env, "_smooth_prev_prev"):
            self.env._smooth_prev_prev = prev.clone()
        accel = curr - 2.0 * prev + self.env._smooth_prev_prev
        self.env._smooth_prev_prev = prev.clone()
        return torch.sum(torch.square(accel), dim=1)

    def _reward_energy(self):
        """Energy penalty: sum of |torque × joint_velocity|.

        能耗惩罚：扭矩绝对值与关节角速度的乘积之和。
        对应赛题 energy 评分项，鼓励高效步态。
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)

    def _reward_energy_score_formula(self):
        """Platform-shaped energy score: exp(-0.01 * mechanical power)."""
        asset = self._get_robot_asset()
        power = torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)
        power = torch.nan_to_num(power, nan=0.0, posinf=1e6, neginf=0.0)
        return torch.exp(-0.01 * power)

    def _reward_pose_score_formula(self):
        """Platform-aligned posture score: exp(-5 * (|roll| + |pitch|))."""
        asset = self._get_robot_asset()
        roll, pitch = self._quat_to_roll_pitch(asset.data.root_quat_w)
        pose_deviation = torch.abs(roll) + torch.abs(pitch)
        pose_deviation = torch.nan_to_num(pose_deviation, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.exp(-5.0 * pose_deviation)

    def _reward_posture_stability(self):
        """Penalize frame-to-frame roll and pitch change for steadier fast gait."""
        asset = self._get_robot_asset()
        roll, pitch = self._quat_to_roll_pitch(asset.data.root_quat_w)
        roll = torch.nan_to_num(roll, nan=0.0, posinf=0.0, neginf=0.0)
        pitch = torch.nan_to_num(pitch, nan=0.0, posinf=0.0, neginf=0.0)

        if not hasattr(self.env, "_posture_prev_roll") or self.env._posture_prev_roll.shape != roll.shape:
            self.env._posture_prev_roll = roll.clone()
            self.env._posture_prev_pitch = pitch.clone()

        roll_rate = torch.abs(roll - self.env._posture_prev_roll)
        pitch_rate = torch.abs(pitch - self.env._posture_prev_pitch)

        try:
            done = self.env.termination_manager.terminated | self.env.termination_manager.time_outs
            if done.any():
                roll_rate[done] = 0.0
                pitch_rate[done] = 0.0
        except Exception:
            pass

        self.env._posture_prev_roll = roll.clone()
        self.env._posture_prev_pitch = pitch.clone()
        return roll_rate + pitch_rate

    def _reward_correct_base_height(self, target_height: float = 0.38):
        """Penalize deviation of base height from target (squared).

        惩罚机身高度偏离目标高度（平方误差）。
        对应赛题 posture 评分项。Go2 标准站立高度 ≈ 0.38 m。

        Args:
            target_height: Target base height in meters. / 目标机身高度（米）。
        """
        asset = self._get_robot_asset()
        return torch.square(asset.data.root_pos_w[:, 2] - target_height)

    def _reward_hip_to_default(self):
        """Penalize hip joint deviation from default angle (squared sum).

        惩罚髋关节偏离默认角度（平方和）。
        Go2 髋关节为每腿第 0 个关节：FL=0, FR=3, RL=6, RR=9。
        辅助保持自然站姿，防止外八或内八步态。
        """
        asset = self._get_robot_asset()
        hip_idx = [0, 3, 6, 9]
        hip_dev = asset.data.joint_pos[:, hip_idx] - asset.data.default_joint_pos[:, hip_idx]
        return torch.sum(torch.square(hip_dev), dim=1)

    # -----------------------------------------------------------------------
    # Gait quality rewards (Round-4 addition)
    # 步态质量奖励（第四轮新增）
    # -----------------------------------------------------------------------

    def _reward_dof_vel(self):
        """Penalize large joint velocities (L2 norm of joint velocity vector).

        惩罚过大的关节速度（关节速度向量的 L2 范数）。
        关节速度无约束是步态"鬼畜"的根本原因之一；该惩罚鼓励流畅、低速摆腿。
        Ref: agent_diy/feature/reward_process.py
        """
        asset = self._get_robot_asset()
        return torch.sum(torch.square(asset.data.joint_vel), dim=1)

    def _reward_base_lateral_vel(self, command_name: str = "base_velocity"):
        """Penalize untracked lateral (Y-axis) velocity to prevent crab walking.

        惩罚**未被指令覆盖**的侧向速度，防止螃蟹步或横向侧滑。

        注意：不能无条件惩罚 root_lin_vel_b[:,1]，因为速度指令中含有 lin_vel_y
        分量（[-0.3, 0.3]），机器人有时被要求侧向行走。无条件惩罚会与
        track_lin_vel_xy 正向奖励产生梯度冲突。

        正确做法：惩罚 (实际侧向速度 - 指令侧向速度)²，即"横移误差"。
        机器人照指令侧走时误差≈0，不罚；无指令自发侧漂时才受惩罚。
        Ref: custom_rewards.py penalize_base_lat_vel_l2 (error-based variant)
        """
        asset = self._get_robot_asset()
        cmd_vy = self._tracking_command(command_name)[:, 1]
        actual_vy = asset.data.root_lin_vel_b[:, 1]
        return torch.square(actual_vy - cmd_vy)

    def _reward_uncommanded_yaw_rate(
        self,
        command_name: str = "base_velocity",
        yaw_cmd_threshold: float = 0.05,
        deadband: float = 0.05,
    ):
        """Penalize yaw rotation when the yaw command is near zero."""
        asset = self._get_robot_asset()
        cmd_yaw = self._tracking_command(command_name)[:, 2]
        yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
        no_turn_cmd = torch.abs(cmd_yaw) < yaw_cmd_threshold
        excess_yaw = torch.clamp(yaw_rate - deadband, min=0.0)
        return no_turn_cmd.float() * torch.square(excess_yaw)

    def _reward_air_time_variance_penalty(self):
        """Penalize variance in per-foot air time to enforce gait rhythm symmetry.

        惩罚各脚滞空时间的方差，促进对称步态节律（如 trot）。
        对称步态更稳定、更节能，有助于提高 posture 与 energy 评分。
        使用 clamp(max=0.5s) 避免单脚异常值主导方差计算。
        Ref: agent_diy/feature/reward_process.py
        """
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        return torch.var(torch.clamp(last_air_time, max=0.5), dim=1)

    def _reward_pivot_turning(
        self,
        lin_vel_threshold: float = 0.2,
        ang_vel_threshold: float = 0.5,
        command_name: str = "base_velocity",
        yaw_cmd_threshold: float = 0.25,
    ):
        """Penalize uncommanded pivoting: rotating on the spot without lifting feet.

        惩罚非指令要求的原地蹭脚转弯——线速度低（< lin_vel_threshold）但偏航角速度高
        （> ang_vel_threshold），且没有明显 yaw 命令时，仍有多脚接触地面则给予惩罚。
        鼓励通过迈步（lift & place）完成转弯，而非用脚蹭地旋转，减少关节磨损和
        步态不自然。
        When the local navigator explicitly commands a large yaw rate, this
        penalty is gated off so emergency in-place turns do not fight the rule
        controller.
        Ref: custom_rewards.py penalize_pivot_turning

        Args:
            lin_vel_threshold: Max horizontal speed (m/s) for pivoting detection.
                               判定原地旋转的最大水平速度阈值 (m/s)。
            ang_vel_threshold: Min yaw rate (rad/s) for pivoting detection.
                               判定原地旋转的最小偏航角速度阈值 (rad/s)。
            command_name: Velocity command name.
            yaw_cmd_threshold: Yaw command above which pivoting is considered commanded.
        """
        asset = self._get_robot_asset()
        sensor_cfg = self._get_foot_sensor_cfg()
        contact_sensor = self.env.scene.sensors[sensor_cfg.name]
        cmd_yaw = torch.abs(self._tracking_command(command_name)[:, 2])

        base_lin_vel = asset.data.root_lin_vel_b
        base_ang_vel = asset.data.root_ang_vel_b

        horizontal_speed = torch.norm(base_lin_vel[:, :2], dim=1)
        commanded_turn = cmd_yaw > yaw_cmd_threshold
        is_pivoting = (horizontal_speed < lin_vel_threshold) & (
            torch.abs(base_ang_vel[:, 2]) > ang_vel_threshold
        ) & ~commanded_turn

        contact_forces = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
        )
        feet_in_contact = contact_forces > 1.0
        num_contacting_feet = torch.sum(feet_in_contact.float(), dim=1)

        return num_contacting_feet * is_pivoting.float()

    # -----------------------------------------------------------------------
    # Goal-reaching rewards (activated only in track terrain)
    # 目标到达奖励（仅 track 地形时激活）
    # -----------------------------------------------------------------------

    def _goal_delta_body(self):
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, 2, device=self.env.device), torch.zeros(
                self.env.num_envs,
                device=self.env.device,
            )

        try:
            robot = self.env.scene["robot"]
            root_pos_w = robot.data.root_pos_w
            quat = robot.data.root_quat_w
        except Exception:
            return torch.zeros(self.env.num_envs, 2, device=self.env.device), torch.zeros(
                self.env.num_envs,
                device=self.env.device,
            )

        delta_w = self.env.goal_positions[:, :2] - root_pos_w[:, :2]
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        heading = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        cos_h = torch.cos(-heading)
        sin_h = torch.sin(-heading)
        local_x = cos_h * delta_w[:, 0] - sin_h * delta_w[:, 1]
        local_y = sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1]
        return torch.stack((local_x, local_y), dim=1), torch.linalg.norm(delta_w, dim=1)

    # -----------------------------------------------------------------------
    # Scan helper — pure height_scanner grid, no navigation.py dependency
    # -----------------------------------------------------------------------

    def _height_grid(self):
        scanner = self.env.scene.sensors.get("height_scanner")
        if scanner is None:
            return None
        scan = scanner.data.pos_w[:, 2:3] - scanner.data.ray_hits_w[..., 2]
        return scan.view(self.env.num_envs, 16, 16)

    def _rough_terrain_gate(
        self,
        body_y_start: int = 4,
        body_y_end: int = 12,
        near_x_start: int = 1,
        near_x_end: int = 10,
        delta_quantile: float = 0.85,
        min_delta: float = 0.025,
        full_delta: float = 0.10,
    ):
        """Soft gate for slopes/stairs using forward height-scan roughness."""
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        window = grid[:, body_y_start:body_y_end, near_x_start:near_x_end]
        if window.shape[1] == 0 or window.shape[2] <= 1:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        deltas = torch.abs(window[:, :, 1:] - window[:, :, :-1])
        if deltas.numel() == 0:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        local_delta = torch.quantile(deltas.flatten(1), delta_quantile, dim=1)
        denom = max(full_delta - min_delta, 1e-6)
        return torch.clamp((local_delta - min_delta) / denom, 0.0, 1.0)

    def _reward_non_maze_flat_route(
        self,
        command_name: str = "base_velocity",
        min_command: float = 0.35,
        body_y_start: int = 1,
        body_y_end: int = 15,
        center_y_start: int = 5,
        center_y_end: int = 11,
        near_x_start: int = 1,
        near_x_end: int = 10,
        roughness_scale: float = 0.11,
        side_advantage_deadband: float = 0.018,
        side_advantage_full: float = 0.09,
        target_lateral_speed: float = 0.32,
        center_lateral_std: float = 0.18,
        target_forward_speed: float = 1.65,
        maze_goal_dist_gate: float = 14.0,
        maze_obstacle_threshold: float = -0.80,
        maze_temperature: float = 0.18,
        maze_wall_gate: float = 0.28,
    ):
        """Encourage non-maze track traversal through flatter local lanes.

        This term does not hard-code "always walk along the edge".  It compares
        left / center / right height-scan lanes.  If a side lane is meaningfully
        flatter than the current center corridor, lateral motion is rewarded; if
        the center is already the best lane, lateral motion is softly discouraged.
        """
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        command = self._tracking_command(command_name)
        moving = torch.linalg.norm(command[:, :2], dim=1) > min_command

        y0 = max(0, min(body_y_start, grid.shape[1] - 1))
        y3 = max(y0 + 1, min(body_y_end, grid.shape[1]))
        yc0 = max(y0 + 1, min(center_y_start, y3 - 1))
        yc1 = max(yc0 + 1, min(center_y_end, y3))
        x0 = max(0, min(near_x_start, grid.shape[2] - 1))
        x1 = max(x0 + 1, min(near_x_end, grid.shape[2]))

        def _lane_cost(window: torch.Tensor) -> torch.Tensor:
            if window.shape[1] == 0 or window.shape[2] == 0:
                return torch.full((self.env.num_envs,), 1.0, device=self.env.device)
            if window.shape[2] > 1:
                forward_delta = torch.abs(window[:, :, 1:] - window[:, :, :-1]).flatten(1)
                forward_cost = torch.quantile(forward_delta, 0.80, dim=1)
            else:
                forward_cost = torch.zeros(self.env.num_envs, device=self.env.device)
            if window.shape[1] > 1:
                lateral_delta = torch.abs(window[:, 1:, :] - window[:, :-1, :]).flatten(1)
                lateral_cost = torch.quantile(lateral_delta, 0.80, dim=1)
            else:
                lateral_cost = torch.zeros(self.env.num_envs, device=self.env.device)
            flat = window.flatten(1)
            spread = torch.quantile(flat, 0.90, dim=1) - torch.quantile(flat, 0.10, dim=1)
            return 0.60 * forward_cost + 0.30 * lateral_cost + 0.10 * spread

        left = grid[:, y0:yc0, x0:x1]
        center = grid[:, yc0:yc1, x0:x1]
        right = grid[:, yc1:y3, x0:x1]
        lane_costs = torch.stack((_lane_cost(left), _lane_cost(center), _lane_cost(right)), dim=1)
        best_cost, best_idx = torch.min(lane_costs, dim=1)
        center_cost = lane_costs[:, 1]

        side_is_better = best_idx != 1
        side_advantage = torch.clamp(
            (center_cost - best_cost - side_advantage_deadband) / max(side_advantage_full, 1e-6),
            0.0,
            1.0,
        ) * side_is_better.float()

        lateral_speed = torch.abs(robot.data.root_lin_vel_b[:, 1])
        shift_score = torch.clamp(lateral_speed / max(target_lateral_speed, 1e-6), 0.0, 1.0)
        center_keep_score = torch.exp(-torch.square(lateral_speed / max(center_lateral_std, 1e-6)))
        lane_action_score = side_advantage * shift_score + (1.0 - side_advantage) * center_keep_score

        flatness_score = torch.exp(-best_cost / max(roughness_scale, 1e-6))
        forward_score = torch.clamp(robot.data.root_lin_vel_b[:, 0] / max(target_forward_speed, 1e-6), 0.0, 1.0)

        front = grid[:, y0:y3, : max(x1, 1)]
        wall_score = torch.sigmoid((maze_obstacle_threshold - front) / max(maze_temperature, 1e-6)).mean(dim=(1, 2))
        try:
            _, goal_dist = self._goal_delta_body()
            maze_gate = (goal_dist < maze_goal_dist_gate).float() * torch.clamp(wall_score / maze_wall_gate, 0.0, 1.0)
        except Exception:
            maze_gate = torch.zeros(self.env.num_envs, device=self.env.device)
        non_maze_gate = 1.0 - torch.clamp(maze_gate, 0.0, 1.0)

        return non_maze_gate * moving.float() * flatness_score * lane_action_score * (0.35 + 0.65 * forward_score)

    def _reward_rough_score_guidance(
        self,
        command_name: str = "base_velocity",
        min_command: float = 0.15,
        tracking_std: float = 0.38,
        posture_weight: float = 0.75,
        body_y_start: int = 4,
        body_y_end: int = 12,
        near_x_start: int = 1,
        near_x_end: int = 10,
        delta_quantile: float = 0.85,
        min_delta: float = 0.025,
        full_delta: float = 0.10,
    ):
        """Official-score-shaped posture/energy bonus on rough terrain."""
        robot = self._get_robot_asset()
        cmd = self._tracking_command(command_name)
        cmd_xy = cmd[:, :2]
        actual_xy = robot.data.root_lin_vel_b[:, :2]
        cmd_speed = torch.linalg.norm(cmd_xy, dim=1)
        moving_cmd = cmd_speed > min_command

        vel_error = torch.sum(torch.square(cmd_xy - actual_xy), dim=1)
        tracking_score = torch.exp(-vel_error / max(tracking_std * tracking_std, 1e-6))

        roll, pitch = self._quat_to_roll_pitch(robot.data.root_quat_w)
        pose_deviation = torch.abs(roll) + torch.abs(pitch)
        pose_deviation = torch.nan_to_num(pose_deviation, nan=0.0, posinf=0.0, neginf=0.0)
        pose_score = torch.exp(-5.0 * pose_deviation)
        power = torch.sum(torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=1)
        energy_score = torch.exp(-0.01 * power)

        rough_gate = self._rough_terrain_gate(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            delta_quantile=delta_quantile,
            min_delta=min_delta,
            full_delta=full_delta,
        )
        posture_weight = min(max(posture_weight, 0.0), 1.0)
        score_hint = posture_weight * pose_score + (1.0 - posture_weight) * energy_score
        return rough_gate * moving_cmd.float() * tracking_score * score_hint

    def _reward_rough_ang_vel_xy(
        self,
        body_y_start: int = 4,
        body_y_end: int = 12,
        near_x_start: int = 1,
        near_x_end: int = 10,
        delta_quantile: float = 0.85,
        min_delta: float = 0.025,
        full_delta: float = 0.10,
    ):
        """Extra pitch/roll angular-velocity penalty only on rough terrain."""
        robot = self._get_robot_asset()
        rough_gate = self._rough_terrain_gate(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            delta_quantile=delta_quantile,
            min_delta=min_delta,
            full_delta=full_delta,
        )
        return rough_gate * torch.linalg.norm(robot.data.root_ang_vel_b[:, :2], dim=1)

    def _reward_rough_roll_pitch_abs(
        self,
        body_y_start: int = 4,
        body_y_end: int = 12,
        near_x_start: int = 1,
        near_x_end: int = 10,
        delta_quantile: float = 0.85,
        min_delta: float = 0.015,
        full_delta: float = 0.075,
    ):
        """Direct official posture penalty amplified on rough terrain."""
        robot = self._get_robot_asset()
        roll, pitch = self._quat_to_roll_pitch(robot.data.root_quat_w)
        pose_deviation = torch.abs(roll) + torch.abs(pitch)
        pose_deviation = torch.nan_to_num(pose_deviation, nan=0.0, posinf=0.0, neginf=0.0)
        rough_gate = self._rough_terrain_gate(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            delta_quantile=delta_quantile,
            min_delta=min_delta,
            full_delta=full_delta,
        )
        return rough_gate * pose_deviation

    def _reward_rough_energy(
        self,
        body_y_start: int = 4,
        body_y_end: int = 12,
        near_x_start: int = 1,
        near_x_end: int = 10,
        delta_quantile: float = 0.85,
        min_delta: float = 0.015,
        full_delta: float = 0.075,
    ):
        """Extra official energy penalty where stairs/slopes usually hurt score."""
        robot = self._get_robot_asset()
        rough_gate = self._rough_terrain_gate(
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            near_x_start=near_x_start,
            near_x_end=near_x_end,
            delta_quantile=delta_quantile,
            min_delta=min_delta,
            full_delta=full_delta,
        )
        power = torch.sum(torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=1)
        return rough_gate * power

    def _goal_vector_body(self):
        local_goal, dist = self._goal_delta_body()
        denom = torch.clamp(dist, min=1e-6).unsqueeze(1)
        goal_dir = local_goal / denom
        return local_goal, dist, goal_dir

    def _wall_score_from_sector(
        self,
        sector: torch.Tensor,
        obstacle_threshold: float = -0.75,
        temperature: float = 0.18,
    ):
        if sector.shape[1] == 0 or sector.shape[2] == 0:
            return torch.zeros(sector.shape[0], device=sector.device)
        return torch.sigmoid((obstacle_threshold - sector) / max(temperature, 1e-6)).mean(dim=(1, 2))

    def _maze_context_gate(
        self,
        grid: torch.Tensor,
        goal_dist_gate: float = 14.0,
        obstacle_threshold: float = -0.80,
        temperature: float = 0.18,
        front_cols: int = 10,
        side_width: int = 3,
        side_col_threshold: float = 0.32,
        side_depth_ratio: float = 0.55,
        front_col_threshold: float = 0.62,
        front_depth_ratio: float = 0.45,
        stair_uniformity_threshold: float = 0.16,
        stair_max_front_depth_ratio: float = 0.32,
    ):
        """Gate maze-only wall rewards away from slopes/stairs.

        Full track is ordered as slopes/stairs first and maze last.  Height scan
        alone is not semantic: stair risers can look like short front walls.  We
        therefore require both a late-track phase (close enough to final goal)
        and a maze-like wall pattern: continuous side walls or a thick front
        blocker.  Thin row-uniform bands are treated as stairs/slopes.
        """
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        num_envs = grid.shape[0]
        cols = max(1, min(int(front_cols), grid.shape[2]))
        side = max(1, min(int(side_width), grid.shape[1] // 2))
        temp = max(temperature, 1e-6)

        wall_prob = torch.sigmoid((obstacle_threshold - grid[:, :, :cols]) / temp)

        left_cols = wall_prob[:, :side, :].mean(dim=1)
        right_cols = wall_prob[:, -side:, :].mean(dim=1)
        left_cont = (left_cols > side_col_threshold).float().mean(dim=1)
        right_cont = (right_cols > side_col_threshold).float().mean(dim=1)
        side_corridor = (left_cont > side_depth_ratio) & (right_cont > side_depth_ratio)

        center = wall_prob[:, side:-side, :] if grid.shape[1] > 2 * side else wall_prob
        center_cols = center.mean(dim=1)
        dense_front_ratio = (center_cols > front_col_threshold).float().mean(dim=1)
        thick_front_wall = dense_front_ratio > front_depth_ratio

        # Stair/slope risers usually span almost the whole track width and only
        # occupy thin depth bands.  Maze walls are laterally localized or thick.
        raw_center = grid[:, side:-side, :cols] if grid.shape[1] > 2 * side else grid[:, :, :cols]
        lateral_uniformity = raw_center.std(dim=1).mean(dim=1)
        stair_or_slope_like = (
            (lateral_uniformity < stair_uniformity_threshold)
            & (dense_front_ratio < stair_max_front_depth_ratio)
        )

        front_gate = thick_front_wall.float() * (1.0 - stair_or_slope_like.float())
        visual_gate = torch.clamp(side_corridor.float() + front_gate, max=1.0)

        if hasattr(self.env, "goal_positions") and self.env.goal_positions is not None:
            _, goal_dist = self._goal_delta_body()
            phase_gate = (goal_dist < goal_dist_gate).float()
        else:
            phase_gate = torch.ones(num_envs, device=grid.device)

        return phase_gate * visual_gate

    def _reward_maze_context_gate(
        self,
        goal_dist_gate: float = 14.0,
        obstacle_threshold: float = -0.80,
        temperature: float = 0.18,
        front_cols: int = 10,
    ):
        """Diagnostic only: 1 when wall rewards are allowed to behave as maze logic."""
        grid = self._height_grid()
        return self._maze_context_gate(
            grid,
            goal_dist_gate=goal_dist_gate,
            obstacle_threshold=obstacle_threshold,
            temperature=temperature,
            front_cols=front_cols,
        )

    def _maze_front_wall_turn_features(
        self,
        obstacle_threshold: float = -0.72,
        temperature: float = 0.18,
        front_cols: int = 6,
        body_y_start: int = 3,
        body_y_end: int = 13,
        side_width: int = 4,
        wall_start: float = 0.28,
        wall_full: float = 0.72,
        maze_goal_dist_gate: float = 14.0,
        maze_gate_obstacle_threshold: float = -0.80,
    ):
        grid = self._height_grid()
        if grid is None:
            zeros = torch.zeros(self.env.num_envs, device=self.env.device)
            return zeros, zeros

        front_cols = max(1, min(int(front_cols), grid.shape[2]))
        body_y_start = max(0, int(body_y_start))
        body_y_end = min(int(body_y_end), grid.shape[1])
        side_width = max(1, min(int(side_width), grid.shape[1] // 2))
        if body_y_end <= body_y_start:
            zeros = torch.zeros(self.env.num_envs, device=self.env.device)
            return zeros, zeros

        wall_prob = torch.sigmoid((obstacle_threshold - grid[:, :, :front_cols]) / max(temperature, 1e-6))
        center_wall = wall_prob[:, body_y_start:body_y_end, :].mean(dim=(1, 2))
        left_open = 1.0 - wall_prob[:, :side_width, :].mean(dim=(1, 2))
        right_open = 1.0 - wall_prob[:, -side_width:, :].mean(dim=(1, 2))
        open_delta = right_open - left_open

        _, _, goal_dir = self._goal_vector_body()
        goal_turn = torch.sign(goal_dir[:, 1])
        visual_turn = torch.sign(open_delta)
        turn_sign = torch.where(torch.abs(open_delta) > 0.08, visual_turn, goal_turn)

        wall_gate = torch.clamp((center_wall - wall_start) / max(wall_full - wall_start, 1e-6), 0.0, 1.0)
        maze_gate = self._maze_context_gate(
            grid,
            goal_dist_gate=maze_goal_dist_gate,
            obstacle_threshold=maze_gate_obstacle_threshold,
            temperature=temperature,
        )
        return wall_gate * maze_gate, turn_sign

    def _reward_maze_anticipatory_turn(
        self,
        obstacle_threshold: float = -0.72,
        temperature: float = 0.18,
        front_cols: int = 6,
        body_y_start: int = 3,
        body_y_end: int = 13,
        side_width: int = 4,
        wall_start: float = 0.28,
        wall_full: float = 0.72,
        target_yaw_rate: float = 0.75,
        target_forward_speed: float = 0.95,
        maze_goal_dist_gate: float = 14.0,
        maze_gate_obstacle_threshold: float = -0.80,
    ):
        """Reward fast arcing turns toward a side opening before a maze wall."""
        robot = self._get_robot_asset()
        wall_gate, turn_sign = self._maze_front_wall_turn_features(
            obstacle_threshold=obstacle_threshold,
            temperature=temperature,
            front_cols=front_cols,
            body_y_start=body_y_start,
            body_y_end=body_y_end,
            side_width=side_width,
            wall_start=wall_start,
            wall_full=wall_full,
            maze_goal_dist_gate=maze_goal_dist_gate,
            maze_gate_obstacle_threshold=maze_gate_obstacle_threshold,
        )
        yaw_toward_opening = torch.clamp(
            robot.data.root_ang_vel_b[:, 2] * turn_sign / max(target_yaw_rate, 1e-6),
            min=0.0,
            max=1.0,
        )
        speed_score = torch.clamp(
            robot.data.root_lin_vel_b[:, 0] / max(target_forward_speed, 1e-6),
            min=0.0,
            max=1.0,
        )
        return wall_gate * yaw_toward_opening * speed_score

    def _reward_forward_heading_velocity(
        self,
        target_speed: float = 0.55,
        max_reward: float = 1.0,
    ):
        """Reward moving forward in the robot head/body direction."""
        robot = self._get_robot_asset()
        vx = robot.data.root_lin_vel_b[:, 0]
        return torch.clamp(vx / max(target_speed, 1e-6), min=0.0, max=max_reward)

    def _reward_backward_penalty(self, deadband: float = 0.03):
        """Penalize walking backward relative to the robot head direction."""
        robot = self._get_robot_asset()
        vx = robot.data.root_lin_vel_b[:, 0]
        return torch.clamp(-(vx + deadband), min=0.0)

    def _reward_goal_heading_alignment(self, std: float = 0.75):
        """Reward facing the target with the head before moving forward."""
        _, dist, goal_dir = self._goal_vector_body()
        # goal_dir[:, 0] = cos(target angle in body frame).
        angle_error = torch.atan2(goal_dir[:, 1], goal_dir[:, 0])
        return torch.exp(-torch.square(angle_error / max(std, 1e-6))) * (dist > 0.6).float()

    def _reward_goal_velocity_projection(self, max_speed: float = 0.75):
        """Reward body velocity projected onto the target direction."""
        robot = self._get_robot_asset()
        _, dist, goal_dir = self._goal_vector_body()
        body_xy = robot.data.root_lin_vel_b[:, :2]
        projection = torch.sum(body_xy * goal_dir, dim=1)
        return torch.clamp(projection / max(max_speed, 1e-6), min=-1.0, max=1.0) * (dist > 0.6).float()

    def _reward_goal_backtrack_penalty(self, deadband: float = 0.02):
        """Penalize velocity whose projection moves away from the target."""
        robot = self._get_robot_asset()
        _, dist, goal_dir = self._goal_vector_body()
        projection = torch.sum(robot.data.root_lin_vel_b[:, :2] * goal_dir, dim=1)
        return torch.clamp(-(projection + deadband), min=0.0) * (dist > 0.8).float()

    def _reward_goal_distance(self, scale: float = 8.0):
        """Dense bounded reward that increases as the robot gets closer."""
        _, dist = self._goal_delta_body()
        return torch.exp(-dist / max(scale, 1e-6))

    def _reward_task_complete(self, threshold: float = 0.6):
        """Large sparse completion reward aligned with the official goal radius."""
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        dist = torch.norm(goal_pos - robot_pos, dim=1)
        return (dist < threshold).float()

    def _reward_wall_proximity(
        self,
        obstacle_threshold: float = -0.55,
        front_cols: int = 7,
        body_y_start: int = 2,
        body_y_end: int = 14,
        wall_score_threshold: float = 0.18,
        temperature: float = 0.18,
        maze_goal_dist_gate: float = 14.0,
        maze_gate_obstacle_threshold: float = -0.80,
    ):
        """Small penalty for being close to wall-like geometry."""
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        sector = grid[:, body_y_start:body_y_end, :front_cols]
        wall_score = self._wall_score_from_sector(sector, obstacle_threshold, temperature)
        gate = self._maze_context_gate(
            grid,
            goal_dist_gate=maze_goal_dist_gate,
            obstacle_threshold=maze_gate_obstacle_threshold,
            temperature=temperature,
        )
        return torch.clamp(wall_score - wall_score_threshold, min=0.0) * gate

    def _reward_wall_collision(
        self,
        obstacle_threshold: float = -0.75,
        front_cols: int = 3,
        body_y_start: int = 3,
        body_y_end: int = 13,
        wall_score_threshold: float = 0.55,
        temperature: float = 0.18,
        touch_penalty: float = 0.12,
        slow_speed: float = 0.15,
        impact_speed: float = 0.55,
        impact_penalty: float = 1.60,
        maze_goal_dist_gate: float = 14.0,
        maze_gate_obstacle_threshold: float = -0.80,
    ):
        """Speed-scaled wall penalty: touching slowly is cheap, ramming is expensive."""
        robot = self._get_robot_asset()
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        sector = grid[:, body_y_start:body_y_end, :front_cols]
        wall_score = self._wall_score_from_sector(sector, obstacle_threshold, temperature)
        forward_speed = torch.clamp(robot.data.root_lin_vel_b[:, 0], min=0.0)
        wall_intensity = torch.clamp(
            (wall_score - wall_score_threshold) / max(1.0 - wall_score_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )
        speed_ratio = torch.clamp(
            (forward_speed - slow_speed) / max(impact_speed - slow_speed, 1e-6),
            min=0.0,
            max=1.0,
        )
        penalty = touch_penalty + (impact_penalty - touch_penalty) * torch.square(speed_ratio)
        gate = self._maze_context_gate(
            grid,
            goal_dist_gate=maze_goal_dist_gate,
            obstacle_threshold=maze_gate_obstacle_threshold,
            temperature=temperature,
        )
        return wall_intensity * penalty * gate

    def _reward_wall_stall_penalty(
        self,
        obstacle_threshold: float = -0.70,
        front_cols: int = 5,
        body_y_start: int = 3,
        body_y_end: int = 13,
        wall_score_threshold: float = 0.38,
        temperature: float = 0.18,
        still_speed: float = 0.12,
        goal_dist_threshold: float = 0.8,
        maze_goal_dist_gate: float = 14.0,
        maze_gate_obstacle_threshold: float = -0.80,
    ):
        """Penalty for waiting near a clear front wall or pillar.

        Slow contact is still allowed by wall_collision. This term only fires
        when the scan sees a strong blocker ahead and the body is barely moving,
        which matches the observed wall/pillar timeout failure mode.
        """
        robot = self._get_robot_asset()
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        front_cols = max(1, min(int(front_cols), grid.shape[2]))
        sector = grid[:, body_y_start:body_y_end, :front_cols]
        wall_score = self._wall_score_from_sector(sector, obstacle_threshold, temperature)
        wall_intensity = torch.clamp(
            (wall_score - wall_score_threshold) / max(1.0 - wall_score_threshold, 1e-6),
            min=0.0,
            max=1.0,
        )

        body_speed = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
        _, goal_dist = self._goal_delta_body()
        stall_gate = (body_speed < still_speed).float() * (goal_dist > goal_dist_threshold).float()
        maze_gate = self._maze_context_gate(
            grid,
            goal_dist_gate=maze_goal_dist_gate,
            obstacle_threshold=maze_gate_obstacle_threshold,
            temperature=temperature,
        )
        return wall_intensity * stall_gate * maze_gate

    def _reward_open_space(
        self,
        obstacle_threshold: float = -0.35,
        front_cols: int = 8,
        body_y_start: int = 1,
        body_y_end: int = 15,
        maze_goal_dist_gate: float = 14.0,
        maze_gate_obstacle_threshold: float = -0.80,
    ):
        """Tiny reward for staying in locally open space."""
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        sector = grid[:, body_y_start:body_y_end, :front_cols]
        gate = self._maze_context_gate(
            grid,
            goal_dist_gate=maze_goal_dist_gate,
            obstacle_threshold=maze_gate_obstacle_threshold,
        )
        return (sector > obstacle_threshold).float().mean(dim=(1, 2)) * gate

    def _reward_corridor_centering(
        self,
        obstacle_threshold: float = -0.55,
        front_cols: int = 8,
        wall_score_threshold: float = 0.20,
        temperature: float = 0.18,
        center_band_half_width: int = 1,
        maze_goal_dist_gate: float = 14.0,
        maze_gate_obstacle_threshold: float = -0.80,
    ):
        """Penalize off-center walking only when both corridor walls are visible."""
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        front_cols = max(1, min(int(front_cols), grid.shape[2]))
        row_wall_score = torch.sigmoid(
            (obstacle_threshold - grid[:, :, :front_cols]) / max(temperature, 1e-6)
        ).mean(dim=2)

        num_rows = row_wall_score.shape[1]
        row_idx = torch.arange(num_rows, device=grid.device, dtype=grid.dtype)
        center = 0.5 * float(num_rows - 1)
        half_width = max(float(center_band_half_width), 0.0)
        left_mask = row_idx < center - half_width
        right_mask = row_idx > center + half_width

        left_score = torch.where(left_mask.unsqueeze(0), row_wall_score, torch.zeros_like(row_wall_score))
        right_score = torch.where(right_mask.unsqueeze(0), row_wall_score, torch.zeros_like(row_wall_score))
        left_strength = left_score.max(dim=1).values
        right_strength = right_score.max(dim=1).values
        corridor_gate = ((left_strength > wall_score_threshold) & (right_strength > wall_score_threshold)).float()

        dist_to_center = torch.abs(row_idx - center).unsqueeze(0)
        left_weight = left_score * left_score
        right_weight = right_score * right_score
        left_dist = torch.sum(left_weight * dist_to_center, dim=1) / torch.clamp(
            left_weight.sum(dim=1), min=1e-6
        )
        right_dist = torch.sum(right_weight * dist_to_center, dim=1) / torch.clamp(
            right_weight.sum(dim=1), min=1e-6
        )
        imbalance = torch.abs(left_dist - right_dist) / torch.clamp(left_dist + right_dist, min=1e-6)
        maze_gate = self._maze_context_gate(
            grid,
            goal_dist_gate=maze_goal_dist_gate,
            obstacle_threshold=maze_gate_obstacle_threshold,
            temperature=temperature,
        )
        return corridor_gate * imbalance * maze_gate

    def _reward_directed_exploration(
        self,
        radius: float = 0.55,
        memory_size: int = 96,
        goal_heading_std: float = 1.0,
    ):
        """Tiny novelty reward, gated by target-facing direction.

        This prevents the policy from getting paid for random wandering away
        from the maze goal.
        """
        robot = self._get_robot_asset()
        pos = robot.data.root_pos_w[:, :2]
        num_envs = self.env.num_envs
        device = self.env.device

        if (
            not hasattr(self.env, "_rl_nav_visit_pos")
            or self.env._rl_nav_visit_pos.shape[0] != num_envs
            or self.env._rl_nav_visit_pos.shape[1] != memory_size
        ):
            self.env._rl_nav_visit_pos = torch.zeros(num_envs, memory_size, 2, device=device)
            self.env._rl_nav_visit_valid = torch.zeros(num_envs, memory_size, dtype=torch.bool, device=device)
            self.env._rl_nav_visit_ptr = torch.zeros(num_envs, dtype=torch.long, device=device)

        visit_pos = self.env._rl_nav_visit_pos
        valid = self.env._rl_nav_visit_valid
        dist_to_seen = torch.linalg.norm(visit_pos - pos.unsqueeze(1), dim=2)
        dist_to_seen = torch.where(valid, dist_to_seen, torch.full_like(dist_to_seen, 1e6))
        novel = dist_to_seen.min(dim=1).values > radius

        _, goal_dist, goal_dir = self._goal_vector_body()
        angle_error = torch.atan2(goal_dir[:, 1], goal_dir[:, 0])
        toward_goal_gate = torch.exp(-torch.square(angle_error / max(goal_heading_std, 1e-6)))
        reward = novel.float() * toward_goal_gate * (goal_dist > 1.0).float()

        ptr = self.env._rl_nav_visit_ptr
        env_ids = torch.arange(num_envs, device=device)
        if novel.any():
            add_ids = env_ids[novel]
            add_ptr = ptr[novel]
            visit_pos[add_ids, add_ptr] = pos[novel]
            valid[add_ids, add_ptr] = True
            ptr[novel] = (add_ptr + 1) % memory_size

        try:
            done = self.env.termination_manager.terminated | self.env.termination_manager.time_outs
            if done.any():
                visit_pos[done] = 0.0
                valid[done] = False
                ptr[done] = 0
        except Exception:
            pass

        return reward

    def _reward_wall_buffer(
        self,
        obstacle_threshold: float = -0.75,
        front_rows: int = 5,
        body_y_start: int = 3,
        body_y_end: int = 13,
        wall_score_threshold: float = 0.35,
        temperature: float = 0.18,
    ):
        """Penalize only clear wall-like front blocks.

        height_scan is not semantic: stairs and slopes also create height
        changes. This term therefore uses a stricter wall score instead of
        punishing every non-flat or narrow patch.
        """
        grid = self._height_grid()
        if grid is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        near = grid[:, body_y_start:body_y_end, :front_rows]
        wall_score = torch.sigmoid((obstacle_threshold - near) / max(temperature, 1e-6)).mean(dim=(1, 2))
        return torch.clamp(wall_score - wall_score_threshold, min=0.0)

    def _reward_forward_free_space(self, obstacle_threshold: float = -0.30, front_rows: int = 6,
                                   body_y_start: int = 3, body_y_end: int = 13):
        """Reward forward velocity scaled by front openness.

        When the path ahead is clear the robot gets full credit for forward
        speed; when blocked the reward is suppressed, discouraging wall-
        hugging that occasionally yields approach_goal progress.

        Pure visual gating — no nav_route_info.
        前方畅通时奖励前进速度，前方堵塞时抑制——纯视觉门控。
        """
        robot = self._get_robot_asset()
        grid = self._height_grid()
        forward_speed = robot.data.root_lin_vel_b[:, 0].clamp(min=0.0)
        if grid is None:
            return forward_speed
        near = grid[:, body_y_start:body_y_end, :front_rows]
        open_score = (near > obstacle_threshold).float().mean(dim=(1, 2))
        return forward_speed * open_score

    def _reward_approach_goal(self):
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        _, current_dist = self._goal_delta_body()
        if (
            not hasattr(self.env, "_nav_previous_goal_dist")
            or self.env._nav_previous_goal_dist.shape != current_dist.shape
        ):
            self.env._nav_previous_goal_dist = current_dist.clone()
            self.env._nav_previous_goal_valid = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self.env.device
            )

        if (
            not hasattr(self.env, "_nav_previous_goal_valid")
            or self.env._nav_previous_goal_valid.shape != current_dist.shape
        ):
            self.env._nav_previous_goal_valid = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self.env.device
            )

        delta = current_dist - self.env._nav_previous_goal_dist
        term_mgr = self.env.termination_manager
        reset_mask = term_mgr.terminated | term_mgr.time_outs
        valid_mask = self.env._nav_previous_goal_valid & ~reset_mask
        delta = torch.where(valid_mask, delta, torch.zeros_like(delta))
        self.env._nav_previous_goal_dist = current_dist.clone()
        self.env._nav_previous_goal_valid = ~reset_mask
        return -delta

    def _reward_stuck_penalty(self, min_command: float = 0.15, still_speed: float = 0.08):
        robot = self._get_robot_asset()
        _, dist = self._goal_delta_body()
        cmd = self._tracking_command("base_velocity")
        cmd_speed = torch.linalg.norm(cmd[:, :2], dim=1)
        body_speed = torch.linalg.norm(robot.data.root_lin_vel_b[:, :2], dim=1)
        return ((cmd_speed > min_command) & (body_speed < still_speed) & (dist > 0.8)).float()

    def _reward_navigation_time(self):
        return torch.ones(self.env.num_envs, device=self.env.device)

    def _reward_reach_goal(self, threshold: float = 0.6):
        """Reward for reaching the maze exit (returns 1.0 when distance < 0.6 m).
        到达迷宫出口奖励（distance < 0.6 m 时返回 1.0）。

        Note:
            The threshold must match the threshold of _goal_reached_termination
            in tools/unitree_rl_lab/.../velocity_env_cfg.py (currently 0.6 m),
            otherwise a "termination-reward dead zone" will appear.
            threshold 必须与 tools/unitree_rl_lab/.../velocity_env_cfg.py 中
            _goal_reached_termination 的 threshold 一致（当前 0.6 m），
            否则会产生"终止-奖励死区"。
        """
        if not hasattr(self.env, "goal_positions") or self.env.goal_positions is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)

        robot = self._get_robot_asset()
        robot_pos = robot.data.root_pos_w[:, :2]
        goal_pos = self.env.goal_positions[:, :2]
        dist = torch.norm(goal_pos - robot_pos, dim=1)
        return (dist < threshold).float()
