# -*- coding: UTF-8 -*-
"""
Motion Primitives Controller for flat-ground maze navigation.

将离散的高层导航意图（直走 / 左转 / 右转 / 停止）转换为连续速度指令序列。
使用闭环反馈（yaw 角、距离）精确执行，并提供"原语是否完成"的状态接口。

设计思路：
  - 上层导航器决定"现在该做什么大动作"（如：左转 90° → 直走 2 格 → 右转 90°）
  - 本控制器负责把动作稳定执行完毕，期间指令不会频繁跳动
  - 只有当前原语完成后，上层才会下发下一个原语
  - 底层 frozen locomotion policy 只需跟踪稳定的 (vx, vy, yaw_rate) 指令
"""

from __future__ import annotations

import math
import torch


def _normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class MotionPrimitiveController:
    """
    运动原语控制器

    原语类型：
      IDLE        : 空闲，等待新指令
      FORWARD     : 朝目标 waypoint 直走（闭环跟踪）
      TURN_LEFT   : 向左转到目标朝向
      TURN_RIGHT  : 向右转到目标朝向
      STOP        : 停止
    """

    IDLE = 0
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    STOP = 4

    def __init__(self, num_envs: int, device, config: dict | None = None):
        cfg = config or {}
        self.num_envs = num_envs
        self.device = device

        # ----- 原语执行参数 -----
        self.forward_speed = float(cfg.get("primitive_forward_speed", 0.6))
        self.turn_forward_speed = float(cfg.get("turn_forward_speed", 0.15))
        self.turn_rate = float(cfg.get("primitive_turn_rate", 1.0))
        self.turn_slow_rate = float(cfg.get("primitive_turn_slow_rate", 0.4))
        self.arrive_threshold = float(cfg.get("primitive_arrive_threshold", 0.15))
        self.turn_tolerance = float(cfg.get("primitive_turn_tolerance", 0.10))
        self.forward_yaw_gain = float(cfg.get("primitive_forward_yaw_gain", 0.8))
        self.cmd_smoothing = float(cfg.get("primitive_cmd_smoothing", 0.3))
        self.max_forward_steps = int(cfg.get("primitive_max_forward_steps", 80))
        self.turn_timeout_steps = int(cfg.get("primitive_turn_timeout_steps", 60))

        # ----- 状态 -----
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.target_yaw = torch.zeros(num_envs, device=device)      # 世界坐标系目标朝向
        self.target_pos = torch.zeros(num_envs, 2, device=device)   # 世界坐标系目标位置
        self.prev_cmd = torch.zeros(num_envs, 3, device=device)
        self.step_counter = torch.zeros(num_envs, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # 上层导航器接口：设置原语
    # ------------------------------------------------------------------
    def set_forward(self, env_ids: torch.Tensor, target_pos: torch.Tensor):
        """
        设置直走原语。
        target_pos: [N, 2] 世界坐标 (x, y)
        """
        self.state[env_ids] = self.FORWARD
        self.target_pos[env_ids] = target_pos
        self.step_counter[env_ids] = 0

    def set_turn_left(self, env_ids: torch.Tensor, target_yaw: torch.Tensor):
        """设置左转原语。target_yaw: [N] 世界坐标系目标朝向（弧度）"""
        self.state[env_ids] = self.TURN_LEFT
        self.target_yaw[env_ids] = target_yaw
        self.step_counter[env_ids] = 0

    def set_turn_right(self, env_ids: torch.Tensor, target_yaw: torch.Tensor):
        """设置右转原语。target_yaw: [N] 世界坐标系目标朝向（弧度）"""
        self.state[env_ids] = self.TURN_RIGHT
        self.target_yaw[env_ids] = target_yaw
        self.step_counter[env_ids] = 0

    def set_stop(self, env_ids: torch.Tensor):
        self.state[env_ids] = self.STOP
        self.step_counter[env_ids] = 0

    def is_idle(self, env_ids: torch.Tensor):
        """返回指定环境是否空闲（可接受新原语）"""
        return self.state[env_ids] == self.IDLE

    # ------------------------------------------------------------------
    # 核心更新：每步调用，生成速度指令
    # ------------------------------------------------------------------
    def update(self, robot_pos: torch.Tensor, robot_yaw: torch.Tensor):
        """
        根据当前原语状态生成速度指令。

        Args:
            robot_pos: [num_envs, 2] 世界坐标
            robot_yaw: [num_envs] 世界坐标系朝向（弧度）

        Returns:
            commands: [num_envs, 3] 速度指令 (vx, vy, yaw_rate)
        """
        commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.step_counter += 1

        # ---- FORWARD: 朝世界坐标目标前进 ----
        fwd_mask = self.state == self.FORWARD
        if fwd_mask.any():
            to_target = self.target_pos[fwd_mask] - robot_pos[fwd_mask]  # [N, 2]
            dist = torch.norm(to_target, dim=1)

            target_angle = torch.atan2(to_target[:, 1], to_target[:, 0])
            angle_diff = target_angle - robot_yaw[fwd_mask]
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

            # 到达检测 or 超时
            arrived = dist < self.arrive_threshold
            timeout = self.step_counter[fwd_mask] > self.max_forward_steps
            done = arrived | timeout
            self.state[fwd_mask] = torch.where(
                done,
                torch.tensor(self.IDLE, device=self.device),
                self.state[fwd_mask],
            )

            # 快到目标时减速
            vx = torch.where(
                dist < self.arrive_threshold * 3,
                dist.clamp(min=0.0) * 0.5,
                torch.full_like(dist, self.forward_speed),
            )
            vx = vx.clamp(0.0, self.forward_speed)

            yaw_rate = (angle_diff * self.forward_yaw_gain).clamp(-1.0, 1.0)
            commands[fwd_mask, 0] = vx
            commands[fwd_mask, 2] = yaw_rate

        # ---- TURN_LEFT / TURN_RIGHT: 转向目标朝向 ----
        turn_mask = (self.state == self.TURN_LEFT) | (self.state == self.TURN_RIGHT)
        if turn_mask.any():
            diff = self.target_yaw[turn_mask] - robot_yaw[turn_mask]
            diff = torch.atan2(torch.sin(diff), torch.cos(diff))

            done = torch.abs(diff) < self.turn_tolerance
            timeout = self.step_counter[turn_mask] > self.turn_timeout_steps
            self.state[turn_mask] = torch.where(
                done | timeout,
                torch.tensor(self.IDLE, device=self.device),
                self.state[turn_mask],
            )

            # 接近目标时减速
            rate = torch.where(
                torch.abs(diff) < self.turn_tolerance * 3,
                torch.full_like(diff, self.turn_slow_rate),
                torch.full_like(diff, self.turn_rate),
            )
            yaw_rate = torch.clamp(diff * 2.0, -rate, rate)
            # 边前进边转弯（vx>0 更接近训练分布，避免原地 pivot turn）
            commands[turn_mask, 0] = self.turn_forward_speed
            commands[turn_mask, 2] = yaw_rate

        # ---- STOP / IDLE ----
        stop_mask = (self.state == self.STOP) | (self.state == self.IDLE)
        commands[stop_mask] = 0.0

        # ---- 指令平滑 ----
        alpha = max(0.0, min(1.0, self.cmd_smoothing))
        commands = alpha * commands + (1.0 - alpha) * self.prev_cmd
        self.prev_cmd = commands.clone()

        return commands

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.state.zero_()
            self.prev_cmd.zero_()
            self.step_counter.zero_()
            self.target_yaw.zero_()
            self.target_pos.zero_()
        else:
            self.state[env_ids] = self.IDLE
            self.prev_cmd[env_ids] = 0.0
            self.step_counter[env_ids] = 0
            self.target_yaw[env_ids] = 0.0
            self.target_pos[env_ids] = 0.0
