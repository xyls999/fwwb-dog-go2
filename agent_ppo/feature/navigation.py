# -*- coding: UTF-8 -*-
"""Rule-based local navigation for track terrain.

Hard-rule planner:
  - Goal direction is the default route.
  - Obstacle safety has higher priority than goal seeking.
  - Within the same safety priority, choose the lane closest to the goal.
  - Forward speed is highest when the safe lane still points toward the goal.
  - A small state machine keeps wall-following consistent instead of choosing
    a brand-new route every frame.
"""

from __future__ import annotations

import torch


class LocalNavigationController:
    """Rule-based local navigator with hard collision avoidance."""

    def __init__(self, num_envs: int, device, config: dict | None = None):
        cfg = config or {}
        self.num_envs = num_envs
        self.device = device

        # ---- observation layout ----
        self.scan_start = int(cfg.get("scan_start", 45))
        self.scan_size = int(cfg.get("scan_size", 256))
        self.goal_start = int(cfg.get("goal_start", 301))
        self.goal_size = int(cfg.get("goal_size", 3))

        # ---- wall detection (height-scan values, lower = taller obstacle) ----
        raw_obstacle_threshold = float(cfg.get("raw_obstacle_threshold", -0.30))
        height_scale = float(cfg.get("height_scale", 2.5))
        self.obstacle_threshold = float(
            cfg.get("obstacle_threshold", raw_obstacle_threshold * height_scale)
        )
        self.wall_threshold = float(
            cfg.get("wall_obstacle_threshold", self.obstacle_threshold - 0.45)
        )
        self.wall_temperature = float(cfg.get("wall_temperature", 0.18))
        self.wall_score_threshold = float(cfg.get("wall_score_threshold", 0.45))
        self.very_tall_wall_threshold = float(
            cfg.get("very_tall_wall_threshold", self.wall_threshold - 0.35)
        )
        self.stair_lateral_std_threshold = float(
            cfg.get("stair_lateral_std_threshold", 0.18)
        )
        self.stair_max_wall_height = float(
            cfg.get("stair_max_wall_height", self.wall_threshold + 0.15)
        )
        self.stair_wall_suppression = float(cfg.get("stair_wall_suppression", 0.20))

        # ---- hard stop: immediate front (~20 cm) ----
        self.hard_stop_cols = int(cfg.get("hard_stop_cols", 3))
        self.hard_stop_row_start = int(cfg.get("hard_stop_row_start", 3))
        self.hard_stop_row_end = int(cfg.get("hard_stop_row_end", 13))

        # ---- lane geometry ----
        self.num_lanes = int(cfg.get("num_lanes", 25))
        self.lane_half_width = float(cfg.get("lane_half_width", 2.4))
        self.lane_lookahead_cols = int(cfg.get("lane_lookahead_cols", 8))
        self.max_yaw = float(cfg.get("max_yaw", 1.05))

        # ---- speed ----
        self.max_speed = float(cfg.get("max_speed", 0.85))
        self.min_forward_speed = float(cfg.get("min_forward_speed", 0.20))
        self.turn_speed = float(cfg.get("turn_speed", 0.55))
        self.rotate_yaw = float(cfg.get("rotate_yaw", 0.85))
        self.rotate_90_speed = float(cfg.get("rotate_90_speed", 0.12))
        self.slow_angle = float(cfg.get("slow_angle", 0.55))
        self.goal_yaw_gain = float(cfg.get("goal_yaw_gain", 1.25))
        self.goal_deadband = float(cfg.get("goal_deadband", 0.08))
        self.goal_hard_turn_angle = float(cfg.get("goal_hard_turn_angle", 1.20))
        self.stop_goal_distance = float(cfg.get("stop_goal_distance", 0.03))

        # ---- smoothing ----
        self.smoothing = float(cfg.get("smoothing", 0.35))

        # ---- route state machine / memory / dead-end blacklist ----
        self.MODE_GOAL = 0
        self.MODE_WALL_LEFT = 1
        self.MODE_WALL_RIGHT = 2
        self.commit_steps = int(cfg.get("commit_steps", 24))
        self.wall_follow_yaw = float(cfg.get("wall_follow_yaw", 0.65))
        self.wall_follow_goal_weight = float(cfg.get("wall_follow_goal_weight", 0.35))
        self.wall_follow_min_steps = int(cfg.get("wall_follow_min_steps", 45))
        self.wall_follow_leave_margin = float(cfg.get("wall_follow_leave_margin", 0.25))
        self.wall_follow_leave_angle = float(cfg.get("wall_follow_leave_angle", 0.45))
        self.wall_follow_max_steps = int(cfg.get("wall_follow_max_steps", 420))
        self.memory_decay = float(cfg.get("memory_decay", 0.97))
        self.memory_penalty = float(cfg.get("memory_penalty", 0.18))
        self.memory_radius = int(cfg.get("memory_radius", 1))
        self.memory_cap = float(cfg.get("memory_cap", 6.0))
        self.best_goal_margin = float(cfg.get("best_goal_margin", 0.03))
        self.dead_end_steps = int(cfg.get("dead_end_steps", 70))
        self.blacklist_duration = int(cfg.get("blacklist_duration", 160))
        self.blacklist_radius = int(cfg.get("blacklist_radius", 2))
        self.blacklist_penalty = float(cfg.get("blacklist_penalty", 6.0))

        # ---- anti-stuck ----
        self.stuck_steps = int(cfg.get("stuck_steps", 40))
        self.min_goal_progress = float(cfg.get("min_goal_progress", 0.0002))
        self.escape_duration = int(cfg.get("escape_duration", 32))
        self.escape_yaw = float(cfg.get("escape_yaw", 0.85))
        self.wall_side_margin = float(cfg.get("wall_side_margin", 0.10))

        # ---- hip safety: prevent shoulder-collapse falls ----
        # Policy proprio layout: ang_vel[0:3], gravity[3:6], command[6:9],
        # dof_pos[9:21], dof_vel[21:33], last_action[33:45].
        # Go2 hip joints are dof_pos offsets 0,3,6,9 -> obs indices 9,12,15,18.
        self.hip_indices = [
            int(x) for x in cfg.get("hip_indices", [9, 12, 15, 18])
        ]
        self.hip_default = float(cfg.get("hip_default", 0.0))
        self.hip_max_deviation = float(cfg.get("hip_max_deviation", 1.0))

        # ---- position-based stuck: rotate 90° if body hasn't moved ----
        self.still_speed_threshold = float(cfg.get("still_speed_threshold", 0.05))
        self.still_steps = int(cfg.get("still_steps", 10))
        self.rotate_90_steps = int(cfg.get("rotate_90_steps", 90))

        # ---- maze mode: once inside the maze, treat everything as wall ----
        # Detects "both left AND right walls seen continuously" → maze entry.
        # Before maze, side walls are intermittent; inside they are permanent.
        self.maze_detect_steps = int(cfg.get("maze_detect_steps", 150))
        self.maze_wall_threshold = float(
            cfg.get("maze_wall_threshold", self.wall_threshold)
        )

        # ---- internal state ----
        self._prev_command = torch.zeros(num_envs, 3, device=device)
        self._tie_bias = self._make_tie_bias(num_envs, device)
        self._lane_centers = None
        self._lane_yaws = None
        self._planner_mode = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._mode_timer = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._wall_follow_side = self._tie_bias.clone()
        self._hit_goal_dist = torch.zeros(num_envs, device=device)
        self._best_goal_dist = torch.zeros(num_envs, device=device)
        self._no_improve_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._lane_memory = torch.zeros(num_envs, self.num_lanes, device=device)
        self._lane_blacklist = torch.zeros(num_envs, self.num_lanes, dtype=torch.long, device=device)
        self._committed_lane_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._commit_timer = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Stuck detector state
        self._prev_goal_dist = torch.zeros(num_envs, device=device)
        self._goal_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._stuck_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._escape_timer = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._escape_direction = self._tie_bias.clone()

        # 90-degree rotation state
        self._still_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._rotate_90_timer = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._rotate_90_direction = torch.zeros(num_envs, device=device)

        # Maze mode state — per-direction cumulative counters (never reset)
        self._maze_mode = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._front_wall_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._left_wall_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._right_wall_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_tie_bias(self, num_envs: int, device):
        env_ids = torch.arange(num_envs, device=device)
        return torch.where(env_ids.remainder(2) == 0, 1.0, -1.0)

    def _init_lanes(self, device):
        yaws = torch.linspace(-self.max_yaw, self.max_yaw, self.num_lanes, device=device)
        centers = 7.5 - (yaws / max(self.max_yaw, 1e-6)) * 7.5
        self._lane_yaws = yaws
        self._lane_centers = centers

    def _ensure_state(self, num_envs: int, device):
        if self._prev_command.shape[0] != num_envs or self._prev_command.device != device:
            self._prev_command = torch.zeros(num_envs, 3, device=device)
            self._tie_bias = self._make_tie_bias(num_envs, device)
            self._planner_mode = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._mode_timer = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._wall_follow_side = self._tie_bias.clone()
            self._hit_goal_dist = torch.zeros(num_envs, device=device)
            self._best_goal_dist = torch.zeros(num_envs, device=device)
            self._no_improve_count = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._lane_memory = torch.zeros(num_envs, self.num_lanes, device=device)
            self._lane_blacklist = torch.zeros(num_envs, self.num_lanes, dtype=torch.long, device=device)
            self._committed_lane_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)
            self._commit_timer = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._prev_goal_dist = torch.zeros(num_envs, device=device)
            self._goal_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._stuck_count = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._escape_timer = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._escape_direction = self._tie_bias.clone()
            self._still_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._rotate_90_timer = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._rotate_90_direction = torch.zeros(num_envs, device=device)
            self._maze_mode = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self._front_wall_count = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._left_wall_count = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._right_wall_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, num_envs: int | None = None, device=None, dones=None):
        if num_envs is not None:
            self.num_envs = num_envs
        if device is not None:
            self.device = device
        if dones is None:
            self._prev_command = torch.zeros(self.num_envs, 3, device=self.device)
            self._tie_bias = self._make_tie_bias(self.num_envs, self.device)
            self._planner_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._mode_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._wall_follow_side = self._tie_bias.clone()
            self._hit_goal_dist = torch.zeros(self.num_envs, device=self.device)
            self._best_goal_dist = torch.zeros(self.num_envs, device=self.device)
            self._no_improve_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._lane_memory = torch.zeros(self.num_envs, self.num_lanes, device=self.device)
            self._lane_blacklist = torch.zeros(
                self.num_envs, self.num_lanes, dtype=torch.long, device=self.device
            )
            self._committed_lane_idx = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
            self._commit_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._prev_goal_dist = torch.zeros(self.num_envs, device=self.device)
            self._goal_initialized = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._stuck_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._escape_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._escape_direction = self._tie_bias.clone()
            self._still_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._rotate_90_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._rotate_90_direction = torch.zeros(self.num_envs, device=self.device)
            self._maze_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._front_wall_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._left_wall_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._right_wall_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            return

        if self._prev_command.shape[0] != dones.shape[0] or self._prev_command.device != dones.device:
            self._ensure_state(dones.shape[0], dones.device)
            return

        done_mask = dones.bool().view(-1)
        if done_mask.any():
            self._prev_command = self._prev_command.clone()
            self._prev_command[done_mask] = 0.0
            self._planner_mode = self._planner_mode.clone()
            self._mode_timer = self._mode_timer.clone()
            self._wall_follow_side = self._wall_follow_side.clone()
            self._hit_goal_dist = self._hit_goal_dist.clone()
            self._best_goal_dist = self._best_goal_dist.clone()
            self._no_improve_count = self._no_improve_count.clone()
            self._lane_memory = self._lane_memory.clone()
            self._lane_blacklist = self._lane_blacklist.clone()
            self._committed_lane_idx = self._committed_lane_idx.clone()
            self._commit_timer = self._commit_timer.clone()
            self._prev_goal_dist = self._prev_goal_dist.clone()
            self._goal_initialized = self._goal_initialized.clone()
            self._stuck_count = self._stuck_count.clone()
            self._escape_timer = self._escape_timer.clone()
            self._escape_direction = self._escape_direction.clone()
            self._still_counter = self._still_counter.clone()
            self._rotate_90_timer = self._rotate_90_timer.clone()
            self._rotate_90_direction = self._rotate_90_direction.clone()
            self._maze_mode = self._maze_mode.clone()
            self._front_wall_count = self._front_wall_count.clone()
            self._left_wall_count = self._left_wall_count.clone()
            self._right_wall_count = self._right_wall_count.clone()
            self._planner_mode[done_mask] = self.MODE_GOAL
            self._mode_timer[done_mask] = 0
            self._wall_follow_side[done_mask] = self._tie_bias[done_mask]
            self._hit_goal_dist[done_mask] = 0.0
            self._best_goal_dist[done_mask] = 0.0
            self._no_improve_count[done_mask] = 0
            self._lane_memory[done_mask] = 0.0
            self._lane_blacklist[done_mask] = 0
            self._committed_lane_idx[done_mask] = -1
            self._commit_timer[done_mask] = 0
            self._prev_goal_dist[done_mask] = 0.0
            self._goal_initialized[done_mask] = False
            self._stuck_count[done_mask] = 0
            self._escape_timer[done_mask] = 0
            self._escape_direction[done_mask] = self._tie_bias[done_mask]
            self._still_counter[done_mask] = 0
            self._rotate_90_timer[done_mask] = 0
            self._rotate_90_direction[done_mask] = 0.0
            self._maze_mode[done_mask] = False
            self._front_wall_count[done_mask] = 0
            self._left_wall_count[done_mask] = 0
            self._right_wall_count[done_mask] = 0

    def _wall_score(self, sector: torch.Tensor, threshold=None) -> torch.Tensor:
        """Sigmoid wall score.  sector values < threshold → score → 1.

        `sector` shape: [B, R, C].  `threshold`: scalar or [B] tensor.
        Returns [B].
        """
        if sector.shape[1] == 0 or sector.shape[2] == 0:
            return torch.zeros(sector.shape[0], device=sector.device)
        t = max(self.wall_temperature, 1e-6)
        th = threshold if threshold is not None else self.wall_threshold
        if isinstance(th, torch.Tensor) and th.dim() >= 1:
            th = th.view(-1, 1, 1)
        return torch.sigmoid((th - sector) / t).mean(dim=(1, 2))

    def _stair_like(self, sector: torch.Tensor) -> torch.Tensor:
        """Detect broad stair/slope-like height changes in the height scan."""
        if sector.shape[1] <= 1 or sector.shape[2] == 0:
            return torch.zeros(sector.shape[0], dtype=torch.bool, device=sector.device)
        lateral_std = sector.std(dim=1, unbiased=False).mean(dim=1)
        min_height = sector.amin(dim=(1, 2))
        uniform_lateral = lateral_std < self.stair_lateral_std_threshold
        not_tall_wall = min_height > self.stair_max_wall_height
        very_tall_wall = min_height < self.very_tall_wall_threshold
        return uniform_lateral & not_tall_wall & ~very_tall_wall

    def _suppress_stair_score(
        self,
        score: torch.Tensor,
        sector: torch.Tensor,
        maze_mode: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stair_like = self._stair_like(sector)
        if maze_mode is not None:
            stair_like = stair_like & ~maze_mode.bool()
        factor = min(max(self.stair_wall_suppression, 0.0), 1.0)
        return torch.where(stair_like, score * factor, score), stair_like.float()

    def _all_lane_wall_scores(self, grid: torch.Tensor, threshold=None, maze_mode=None):
        """Wall scores for all lanes → (near_walls [B,L], look_walls [B,L])."""
        if self._lane_centers is None or self._lane_centers.device != grid.device:
            self._init_lanes(grid.device)

        near_list, look_list, stair_list = [], [], []
        for c in self._lane_centers:
            c_val = c.item()
            r_start = max(0, int(torch.floor(torch.tensor(c_val - self.lane_half_width)).item()))
            r_end = min(16, int(torch.ceil(torch.tensor(c_val + self.lane_half_width)).item()) + 1)
            if r_end <= r_start:
                r_end = min(16, r_start + 1)
            near_sector = grid[:, r_start:r_end, : self.hard_stop_cols]
            look_sector = grid[:, r_start:r_end, : self.lane_lookahead_cols]
            near_score = self._wall_score(near_sector, threshold)
            look_score = self._wall_score(look_sector, threshold)
            near_score, near_stair = self._suppress_stair_score(
                near_score, near_sector, maze_mode
            )
            look_score, look_stair = self._suppress_stair_score(
                look_score, look_sector, maze_mode
            )
            near_list.append(near_score)
            look_list.append(look_score)
            stair_list.append(torch.maximum(near_stair, look_stair))
        return (
            torch.stack(near_list, dim=1),
            torch.stack(look_list, dim=1),
            torch.stack(stair_list, dim=1),
        )

    # ------------------------------------------------------------------
    # Main compute
    # ------------------------------------------------------------------

    def compute(
        self, obs: torch.Tensor, update_state: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(command [B,3], stats)`` with vx, vy, yaw_rate."""
        batch = obs.shape[0]
        device = obs.device

        if obs.shape[-1] < self.goal_start + self.goal_size:
            cmd = torch.zeros(batch, 3, device=device, dtype=obs.dtype)
            if update_state:
                self._prev_command = cmd.detach()
            return cmd, {}

        self._ensure_state(batch, device)

        # ---- 1. parse observation ----
        scan = obs[:, self.scan_start : self.scan_start + self.scan_size].reshape(
            batch, 16, 16
        )
        goal = obs[:, self.goal_start : self.goal_start + self.goal_size]
        goal_x, goal_y, goal_dist = goal[:, 0], goal[:, 1], goal[:, 2]
        goal_norm = torch.linalg.norm(goal[:, :2], dim=1)
        goal_angle = torch.atan2(goal_y, goal_x)
        goal_angle = torch.where(goal_norm > 0.03, goal_angle, torch.zeros_like(goal_angle))
        target_yaw = torch.clamp(goal_angle * self.goal_yaw_gain, -self.max_yaw, self.max_yaw)
        target_yaw = torch.where(
            torch.abs(goal_angle) < self.goal_deadband,
            torch.zeros_like(target_yaw),
            target_yaw,
        )

        goal_progress = torch.where(
            self._goal_initialized,
            self._prev_goal_dist - goal_dist,
            torch.zeros_like(goal_dist),
        )
        progress_stalled = (
            self._goal_initialized
            & (goal_dist > self.stop_goal_distance)
            & (goal_progress < self.min_goal_progress)
        )

        # ---- 2. hip safety check ----
        hip_pos = obs[:, self.hip_indices]  # [B, 4]
        hip_dev = torch.abs(hip_pos - self.hip_default)  # [B, 4]
        hip_unsafe = hip_dev.max(dim=1).values > self.hip_max_deviation  # [B]

        # ---- 3. progress-based stuck: policy obs has no base_lin_vel ----
        is_still = progress_stalled

        # ---- 4. countdown existing 90° rotation timer ----
        self._rotate_90_timer = self._rotate_90_timer.clone()
        self._rotate_90_timer = torch.clamp(self._rotate_90_timer - 1, min=0)

        # ---- 5. determine if a new 90° rotation is needed ----
        # Update still counter
        self._still_counter = self._still_counter.clone()
        if update_state:
            self._still_counter = torch.where(
                is_still,
                self._still_counter + 1,
                torch.zeros_like(self._still_counter),
            )
        still_stuck = self._still_counter >= max(self.still_steps, 1)

        need_rotate_90 = hip_unsafe | still_stuck  # [B]

        # 90° rotation direction: toward goal, or random if goal is ahead
        goal_ahead = torch.abs(goal_angle) < 0.15  # ~8.6° deadband
        goal_left_for_rot = goal_angle >= 0.15
        random_dir = self._tie_bias[:batch]
        new_rot_dir = torch.where(
            goal_ahead,
            random_dir,                               # random
            torch.where(
                goal_left_for_rot,
                torch.ones_like(goal_angle),           # turn left (+)
                -torch.ones_like(goal_angle),          # turn right (-)
            ),
        )

        # Enter rotation phase
        self._rotate_90_direction = self._rotate_90_direction.clone()
        if update_state:
            enter_rot = need_rotate_90 & (self._rotate_90_timer <= 0)
            self._rotate_90_timer = torch.where(
                enter_rot,
                torch.full_like(self._rotate_90_timer, max(self.rotate_90_steps, 1)),
                self._rotate_90_timer,
            )
            self._rotate_90_direction = torch.where(
                enter_rot, new_rot_dir, self._rotate_90_direction
            )
            # Reset still counter when rotation starts
            self._still_counter = torch.where(
                enter_rot, torch.zeros_like(self._still_counter), self._still_counter
            )

        rotating_90 = self._rotate_90_timer > 0  # [B]

        # ---- 6. build effective wall threshold (maze_mode persists from prev step) ----
        effective_wall_threshold = torch.where(
            self._maze_mode,
            torch.full((batch,), self.maze_wall_threshold, device=device),
            torch.full((batch,), self.wall_threshold, device=device),
        )

        # ---- 7. hard collision: immediate front ----
        front_near = scan[
            :, self.hard_stop_row_start : self.hard_stop_row_end, : self.hard_stop_cols
        ]
        front_wall_raw = self._wall_score(front_near, threshold=effective_wall_threshold)
        front_wall, front_stair_like = self._suppress_stair_score(
            front_wall_raw, front_near, self._maze_mode
        )
        imminent = front_wall > self.wall_score_threshold

        # ---- 8. side walls (for rotation decision) ----
        left_sector = scan[:, 0:7, : self.hard_stop_cols]
        right_sector = scan[:, 9:16, : self.hard_stop_cols]
        left_near_raw = self._wall_score(left_sector, threshold=effective_wall_threshold)
        right_near_raw = self._wall_score(right_sector, threshold=effective_wall_threshold)
        left_near_wall, left_stair_like = self._suppress_stair_score(
            left_near_raw, left_sector, self._maze_mode
        )
        right_near_wall, right_stair_like = self._suppress_stair_score(
            right_near_raw, right_sector, self._maze_mode
        )

        # ---- 8.5 maze detection: 2+ directions seen walls → maze ----
        # Per-direction cumulative counters (never reset).  Wall-following
        # only hits one direction; a real maze hits 2+ (front+left, left+right, etc.)
        front_seen = front_wall > self.wall_score_threshold
        left_seen = left_near_wall > self.wall_score_threshold
        right_seen = right_near_wall > self.wall_score_threshold
        self._front_wall_count = self._front_wall_count.clone()
        self._left_wall_count = self._left_wall_count.clone()
        self._right_wall_count = self._right_wall_count.clone()
        per_dir_threshold = max(self.maze_detect_steps // 2, 1)
        if update_state:
            self._front_wall_count = torch.where(
                front_seen,
                torch.minimum(self._front_wall_count + 1, torch.full_like(self._front_wall_count, per_dir_threshold + 1)),
                self._front_wall_count,
            )
            self._left_wall_count = torch.where(
                left_seen,
                torch.minimum(self._left_wall_count + 1, torch.full_like(self._left_wall_count, per_dir_threshold + 1)),
                self._left_wall_count,
            )
            self._right_wall_count = torch.where(
                right_seen,
                torch.minimum(self._right_wall_count + 1, torch.full_like(self._right_wall_count, per_dir_threshold + 1)),
                self._right_wall_count,
            )
            dirs_with_wall = (
                (self._front_wall_count >= per_dir_threshold).long()
                + (self._left_wall_count >= per_dir_threshold).long()
                + (self._right_wall_count >= per_dir_threshold).long()
            )
            self._maze_mode = self._maze_mode.clone()
            self._maze_mode = self._maze_mode | (dirs_with_wall >= 2)

        # ---- 9. all-lane wall scores (uses effective threshold) ----
        near_walls, look_walls, lane_stair_like = self._all_lane_wall_scores(
            scan, threshold=effective_wall_threshold, maze_mode=self._maze_mode
        )
        near_blocked = near_walls > self.wall_score_threshold
        look_blocked = look_walls > self.wall_score_threshold
        clear_lanes = ~(near_blocked | look_blocked)
        all_blocked = ~clear_lanes.any(dim=1)  # [B]

        # ---- 10. state machine + safety priority + goal-direction tie-break ----
        yaws = self._lane_yaws  # [L]
        target_error = torch.abs(yaws.unsqueeze(0) - target_yaw.unsqueeze(1))
        goal_lane_idx = torch.argmin(target_error, dim=1)
        batch_idx = torch.arange(batch, device=device)
        goal_lane_blocked = ~clear_lanes[batch_idx, goal_lane_idx]

        # Rotation sign: safety first, then prefer goal side.  Positive yaw
        # follows local +Y / positive goal_angle, i.e. turn left.
        turn_left_clear = right_near_wall > left_near_wall + self.wall_side_margin
        turn_right_clear = left_near_wall > right_near_wall + self.wall_side_margin
        goal_left = goal_angle > self.goal_deadband
        goal_right = goal_angle < -self.goal_deadband
        rotate_sign = torch.where(
            turn_left_clear,
            torch.ones_like(left_near_wall),             # wall on right -> turn left
            torch.where(
                turn_right_clear,
                -torch.ones_like(left_near_wall),        # wall on left -> turn right
                torch.where(
                    goal_left,
                    torch.ones_like(left_near_wall),     # tie: goal on left -> turn left
                    torch.where(
                        goal_right,
                        -torch.ones_like(left_near_wall),# tie: goal on right -> turn right
                        self._tie_bias[:batch],          # goal ahead → random
                    ),
                ),
            ),
        )

        if update_state:
            self._lane_blacklist = torch.clamp(self._lane_blacklist - 1, min=0)
            self._commit_timer = torch.clamp(self._commit_timer - 1, min=0)

            in_wall_mode = self._planner_mode != self.MODE_GOAL
            self._mode_timer = torch.where(
                in_wall_mode,
                self._mode_timer + 1,
                torch.zeros_like(self._mode_timer),
            )

            enough_wall_follow = self._mode_timer >= max(self.wall_follow_min_steps, 1)
            closer_than_hit = goal_dist < (self._hit_goal_dist - self.wall_follow_leave_margin)
            goal_visible = (
                ~goal_lane_blocked
                & ~imminent
                & ~all_blocked
                & (torch.abs(goal_angle) < self.wall_follow_leave_angle)
            )
            leave_wall = in_wall_mode & enough_wall_follow & goal_visible & closer_than_hit
            force_leave = in_wall_mode & (
                self._mode_timer >= max(self.wall_follow_max_steps, self.wall_follow_min_steps)
            )

            self._planner_mode = torch.where(
                leave_wall | force_leave,
                torch.full_like(self._planner_mode, self.MODE_GOAL),
                self._planner_mode,
            )
            self._mode_timer = torch.where(
                leave_wall | force_leave,
                torch.zeros_like(self._mode_timer),
                self._mode_timer,
            )

            enter_wall = (
                (self._planner_mode == self.MODE_GOAL)
                & (goal_lane_blocked | imminent | all_blocked)
                & (goal_dist > self.stop_goal_distance)
            )
            enter_side = torch.where(rotate_sign >= 0.0, 1.0, -1.0)
            enter_mode = torch.where(
                enter_side >= 0.0,
                torch.full_like(self._planner_mode, self.MODE_WALL_LEFT),
                torch.full_like(self._planner_mode, self.MODE_WALL_RIGHT),
            )
            self._planner_mode = torch.where(enter_wall, enter_mode, self._planner_mode)
            self._wall_follow_side = torch.where(
                enter_wall, enter_side, self._wall_follow_side
            )
            self._hit_goal_dist = torch.where(
                enter_wall, goal_dist, self._hit_goal_dist
            )
            self._mode_timer = torch.where(
                enter_wall, torch.zeros_like(self._mode_timer), self._mode_timer
            )
            self._committed_lane_idx = torch.where(
                enter_wall,
                torch.full_like(self._committed_lane_idx, -1),
                self._committed_lane_idx,
            )
            self._commit_timer = torch.where(
                enter_wall, torch.zeros_like(self._commit_timer), self._commit_timer
            )

        wall_follow_active = self._planner_mode != self.MODE_GOAL
        wall_target_yaw = torch.clamp(
            self._wall_follow_side * self.wall_follow_yaw,
            -self.max_yaw,
            self.max_yaw,
        )
        wall_target_error = torch.abs(yaws.unsqueeze(0) - wall_target_yaw.unsqueeze(1))
        goal_w = min(max(self.wall_follow_goal_weight, 0.0), 1.0)
        wall_follow_error = goal_w * target_error + (1.0 - goal_w) * wall_target_error
        route_error = torch.where(
            wall_follow_active.unsqueeze(1),
            wall_follow_error,
            target_error,
        )

        memory_decay = min(max(self.memory_decay, 0.0), 1.0)
        if update_state:
            self._lane_memory = torch.clamp(
                self._lane_memory * memory_decay,
                min=0.0,
                max=self.memory_cap,
            )

        safety_rank = near_blocked.float() * 2.0 + look_blocked.float()
        memory_gate = (
            wall_follow_active | progress_stalled | goal_lane_blocked | imminent | all_blocked
        ).float().unsqueeze(1)
        memory_cost = self._lane_memory * self.memory_penalty * memory_gate
        blacklist_cost = (self._lane_blacklist > 0).float() * self.blacklist_penalty
        lane_cost = (
            safety_rank * 100.0
            + route_error / max(self.max_yaw, 1e-6)
            + memory_cost
            + blacklist_cost
        )

        raw_best_idx = torch.argmin(lane_cost, dim=1)

        committed_idx = torch.clamp(self._committed_lane_idx, min=0, max=self.num_lanes - 1)
        committed_clear = clear_lanes[batch_idx, committed_idx]
        committed_unblocked = self._lane_blacklist[batch_idx, committed_idx] <= 0
        commit_active = (
            (self._commit_timer > 0)
            & (self._committed_lane_idx >= 0)
            & committed_clear
            & committed_unblocked
            & ~imminent
            & ~all_blocked
        )
        best_idx = torch.where(commit_active, committed_idx, raw_best_idx)
        best_yaw = yaws[best_idx]

        best_target_error = target_error[batch_idx, best_idx]
        route_align = torch.clamp(
            1.0 - best_target_error / max(self.max_yaw, 1e-6),
            min=0.0,
            max=1.0,
        )
        route_align = torch.where(
            torch.abs(goal_angle) > self.goal_hard_turn_angle,
            route_align * 0.75,
            route_align,
        )

        best_goal_dist = torch.where(
            self._goal_initialized,
            self._best_goal_dist,
            goal_dist,
        )
        improved_best = goal_dist < (best_goal_dist - self.best_goal_margin)
        bad_route_progress = (
            self._goal_initialized
            & (goal_dist > self.stop_goal_distance)
            & ~improved_best
            & (wall_follow_active | goal_lane_blocked | imminent | all_blocked)
        )
        dead_end_event = torch.zeros(batch, dtype=torch.bool, device=device)
        if update_state:
            self._no_improve_count = torch.where(
                bad_route_progress,
                self._no_improve_count + 1,
                torch.zeros_like(self._no_improve_count),
            )
            dead_end_event = self._no_improve_count >= max(self.dead_end_steps, 1)

            lane_ids = torch.arange(self.num_lanes, device=device).unsqueeze(0)
            selected_dist = torch.abs(lane_ids - best_idx.unsqueeze(1))
            blacklist_mask = dead_end_event.unsqueeze(1) & (
                selected_dist <= max(self.blacklist_radius, 0)
            )
            self._lane_blacklist = torch.where(
                blacklist_mask,
                torch.full_like(self._lane_blacklist, max(self.blacklist_duration, 1)),
                self._lane_blacklist,
            )

            memory_mask = selected_dist <= max(self.memory_radius, 0)
            self._lane_memory = torch.where(
                memory_mask,
                torch.clamp(self._lane_memory + 1.0, max=self.memory_cap),
                self._lane_memory,
            )

            switched_side = torch.where(
                self._wall_follow_side == 0.0,
                torch.where(rotate_sign >= 0.0, -1.0, 1.0),
                -self._wall_follow_side,
            )
            switched_mode = torch.where(
                switched_side >= 0.0,
                torch.full_like(self._planner_mode, self.MODE_WALL_LEFT),
                torch.full_like(self._planner_mode, self.MODE_WALL_RIGHT),
            )
            self._wall_follow_side = torch.where(
                dead_end_event, switched_side, self._wall_follow_side
            )
            self._planner_mode = torch.where(
                dead_end_event, switched_mode, self._planner_mode
            )
            self._hit_goal_dist = torch.where(
                dead_end_event, goal_dist, self._hit_goal_dist
            )
            self._mode_timer = torch.where(
                dead_end_event, torch.zeros_like(self._mode_timer), self._mode_timer
            )
            self._no_improve_count = torch.where(
                dead_end_event, torch.zeros_like(self._no_improve_count), self._no_improve_count
            )
            self._committed_lane_idx = torch.where(
                dead_end_event,
                torch.full_like(self._committed_lane_idx, -1),
                self._committed_lane_idx,
            )
            self._commit_timer = torch.where(
                dead_end_event, torch.zeros_like(self._commit_timer), self._commit_timer
            )

            can_commit = clear_lanes[batch_idx, best_idx] & ~imminent & ~all_blocked
            new_commit = can_commit & (
                (self._committed_lane_idx != best_idx) | (self._commit_timer <= 0)
            )
            self._committed_lane_idx = torch.where(
                new_commit, best_idx, self._committed_lane_idx
            )
            self._commit_timer = torch.where(
                new_commit,
                torch.full_like(self._commit_timer, max(self.commit_steps, 1)),
                self._commit_timer,
            )

            self._best_goal_dist = torch.where(
                self._goal_initialized,
                torch.minimum(self._best_goal_dist, goal_dist),
                goal_dist,
            )

        # ---- 12. wall-based stuck → escape ----
        stuck_candidate = (
            (goal_dist > self.stop_goal_distance)
            & (goal_progress < self.min_goal_progress)
            & (imminent | all_blocked)
            & self._goal_initialized
        )
        self._escape_timer = self._escape_timer.clone()
        self._escape_timer = torch.clamp(self._escape_timer - 1, min=0)

        if update_state:
            stuck_count = self._stuck_count.clone()
            stuck_count = torch.where(
                stuck_candidate & (self._escape_timer <= 0),
                stuck_count + 1,
                torch.zeros_like(stuck_count),
            )
            enter_escape = (stuck_count >= max(self.stuck_steps, 1)) & (self._escape_timer <= 0)
            self._escape_timer = torch.where(
                enter_escape,
                torch.full_like(self._escape_timer, max(self.escape_duration, 1)),
                self._escape_timer,
            )
            self._escape_direction = torch.where(
                enter_escape, rotate_sign, self._escape_direction
            )
            self._stuck_count = torch.where(
                enter_escape, torch.zeros_like(stuck_count), stuck_count
            )

        escaping = self._escape_timer > 0

        # ---- 13. assemble desired yaw & speed  (priority order) ----

        # Base: normal navigation
        desired_yaw = best_yaw.clone()
        desired_speed = self.min_forward_speed + (
            self.max_speed - self.min_forward_speed
        ) * route_align

        # Level 4: imminent collision → stop + rotate (unless already rotating 90)
        desired_yaw = torch.where(
            imminent & ~rotating_90, rotate_sign * self.rotate_yaw, desired_yaw
        )
        desired_speed = torch.where(
            imminent & ~rotating_90, torch.zeros_like(desired_speed), desired_speed
        )

        # Level 3: all blocked → stop + rotate
        desired_yaw = torch.where(
            all_blocked & ~imminent & ~rotating_90,
            rotate_sign * self.rotate_yaw,
            desired_yaw,
        )
        desired_speed = torch.where(
            all_blocked & ~imminent & ~rotating_90,
            torch.zeros_like(desired_speed),
            desired_speed,
        )

        # Level 2: wall-based escape
        desired_yaw = torch.where(
            escaping & ~rotating_90,
            self._escape_direction * self.escape_yaw,
            desired_yaw,
        )

        # Level 1 (highest): 90° rotation — hip unsafe or position stuck
        rot_90_yaw = self._rotate_90_direction * self.rotate_yaw
        desired_yaw = torch.where(rotating_90, rot_90_yaw, desired_yaw)
        desired_speed = torch.where(
            rotating_90,
            torch.full_like(desired_speed, self.rotate_90_speed),
            desired_speed,
        )

        # Slow down for sharp turns (normal mode only)
        hard_turn = torch.abs(desired_yaw) > self.slow_angle
        desired_speed = torch.where(
            hard_turn & ~imminent & ~all_blocked & ~escaping & ~rotating_90,
            torch.minimum(desired_speed, torch.full_like(desired_speed, self.turn_speed)),
            desired_speed,
        )

        # Stop at goal
        at_goal = goal_dist < self.stop_goal_distance
        desired_speed = torch.where(at_goal, torch.zeros_like(desired_speed), desired_speed)
        desired_yaw = torch.where(at_goal, torch.zeros_like(desired_yaw), desired_yaw)

        # ---- 14. smooth & output ----
        command = torch.stack(
            [desired_speed, torch.zeros_like(desired_speed), desired_yaw], dim=1
        )
        alpha = min(max(self.smoothing, 0.0), 1.0)
        command = alpha * command + (1.0 - alpha) * self._prev_command

        if update_state:
            self._prev_command = command.detach()
            self._prev_goal_dist = goal_dist.detach()
            self._goal_initialized = torch.ones_like(self._goal_initialized, dtype=torch.bool)

        stats = {
            "nav_front_wall": front_wall.detach(),
            "nav_front_wall_raw": front_wall_raw.detach(),
            "nav_front_stair_like": front_stair_like.detach(),
            "nav_imminent": imminent.float().detach(),
            "nav_target_yaw": target_yaw.detach(),
            "nav_route_yaw": best_yaw.detach(),
            "nav_best_yaw": best_yaw.detach(),
            "nav_route_align": route_align.detach(),
            "nav_planner_mode": self._planner_mode.float().detach(),
            "nav_wall_follow": wall_follow_active.float().detach(),
            "nav_wall_side": self._wall_follow_side.detach(),
            "nav_mode_timer": self._mode_timer.float().detach(),
            "nav_dead_end": dead_end_event.float().detach(),
            "nav_no_improve": self._no_improve_count.float().detach(),
            "nav_blacklist_count": (self._lane_blacklist > 0).float().sum(dim=1).detach(),
            "nav_memory_mean": self._lane_memory.mean(dim=1).detach(),
            "nav_commit_timer": self._commit_timer.float().detach(),
            "nav_all_blocked": all_blocked.float().detach(),
            "nav_escaping": escaping.float().detach(),
            "nav_hip_unsafe": hip_unsafe.float().detach(),
            "nav_still_stuck": still_stuck.float().detach(),
            "nav_rotating_90": rotating_90.float().detach(),
            "nav_left_wall": left_near_wall.detach(),
            "nav_right_wall": right_near_wall.detach(),
            "nav_left_stair_like": left_stair_like.detach(),
            "nav_right_stair_like": right_stair_like.detach(),
            "nav_lane_stair_like": lane_stair_like.mean(dim=1).detach(),
            "nav_goal_lane_blocked": goal_lane_blocked.float().detach(),
            "nav_goal_progress": goal_progress.detach(),
            "nav_maze_mode": self._maze_mode.float().detach(),
            "nav_maze_f_cnt": self._front_wall_count.float().detach(),
            "nav_maze_l_cnt": self._left_wall_count.float().detach(),
            "nav_maze_r_cnt": self._right_wall_count.float().detach(),
            "nav_cmd_vx": command[:, 0].detach(),
            "nav_cmd_yaw": command[:, 2].detach(),
            "nav_goal_angle": goal_angle.detach(),
            "nav_goal_dist": goal_dist.detach(),
        }
        return command, stats
