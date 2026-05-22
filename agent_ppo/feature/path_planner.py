# -*- coding: UTF-8 -*-
"""
Hybrid Path Planner — Global A* + Local Dynamic Obstacle Avoidance.

全局+局部混合路径规划器：
  - 全局规划器在 height_scan 网格上用 A* 规划到目标方向的最短路径
  - 局部规划器基于全局路径 + 实时障碍物进行动态调整
  - 导航命令基于局部路径生成，带时间平滑
  - 提供路径跟随奖励计算接口
"""

from __future__ import annotations

import heapq
import math
import numpy as np
import torch


def astar_single(grid_free: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
    """
    A* pathfinding on a 2D grid.

    Args:
        grid_free: (H, W) bool numpy array, True = free cell, False = obstacle.
        start: (y, x) tuple.
        goal: (y, x) tuple.

    Returns:
        List of (y, x) tuples from start to goal, empty list if no path.
    """
    H, W = grid_free.shape

    # Clamp to bounds
    start = (max(0, min(H - 1, start[0])), max(0, min(W - 1, start[1])))
    goal = (max(0, min(H - 1, goal[0])), max(0, min(W - 1, goal[1])))

    if not grid_free[start[0], start[1]] or not grid_free[goal[0], goal[1]]:
        # If goal is blocked, try to find nearest free cell
        best = None
        best_dist = float("inf")
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                ny, nx = goal[0] + dy, goal[1] + dx
                if 0 <= ny < H and 0 <= nx < W and grid_free[ny, nx]:
                    d = abs(ny - goal[0]) + abs(nx - goal[1])
                    if d < best_dist:
                        best_dist = d
                        best = (ny, nx)
        if best is None:
            return []
        goal = best

    open_set = [(0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0.0}
    f_score = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + dy, current[1] + dx)
            if 0 <= neighbor[0] < H and 0 <= neighbor[1] < W and grid_free[neighbor[0], neighbor[1]]:
                cost = 1.41421356 if dy != 0 and dx != 0 else 1.0
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []


def line_blocked(grid_obs: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> bool:
    """Bresenham line algorithm to check if line from (y0,x0) to (y1,x1) hits any obstacle."""
    H, W = grid_obs.shape
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy

    while True:
        if 0 <= y0 < H and 0 <= x0 < W and grid_obs[y0, x0]:
            return True
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return False


def inflate_obstacles(grid_obs: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Inflate obstacles by marking neighboring cells as obstacles.
    Uses scipy.ndimage.binary_dilation when available (~20x faster than pure Python).
    """
    try:
        from scipy import ndimage
        return ndimage.binary_dilation(grid_obs, iterations=radius)
    except Exception:
        # Fallback to pure Python
        H, W = grid_obs.shape
        inflated = grid_obs.copy()
        for r in range(H):
            for c in range(W):
                if grid_obs[r, c]:
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                inflated[nr, nc] = True
        return inflated


class HybridPathPlanner:
    """
    Global + Local hybrid path planner on height_scan grid.

    Global path: A* from robot to goal direction on the full height_scan grid.
    Local path:  first N waypoints of global path, with local replanning if blocked.
    """

    def __init__(self, num_envs: int, device, config: dict | None = None):
        cfg = config or {}
        self.num_envs = num_envs
        self.device = device

        # Planning frequencies (steps)
        # Planning frequencies (steps).
        # static_maze: if True, global A* runs ONLY ONCE per episode (at reset),
        # because maze walls are static. This removes ~99% of A* CPU overhead.
        self.static_maze = bool(cfg.get("static_maze", False))
        self.global_plan_steps = int(cfg.get("global_plan_steps", 48))
        self.local_plan_steps = int(cfg.get("local_plan_steps", 12))

        # Grid parameters (height_scan is 16x16)
        self.grid_h = 16
        self.grid_w = 16
        self.resolution = 0.1  # meters per cell

        # Obstacle threshold: height_scan value below this is obstacle
        self.obstacle_threshold = float(cfg.get("obstacle_threshold", -0.30))
        # Inflate obstacles by this many cells to keep path away from edges
        self.inflation_radius = int(cfg.get("inflation_radius", 1))

        # Path parameters
        self.local_path_length = int(cfg.get("local_path_length", 10))
        self.path_follow_tolerance = float(cfg.get("path_follow_tolerance", 0.35))
        self.global_path_max_length = int(cfg.get("global_path_max_length", 30))
        self.lookahead_dist = float(cfg.get("lookahead_dist", 0.8))
        self.min_target_dist = float(cfg.get("min_target_dist", 0.30))
        self.max_target_dist = float(cfg.get("max_target_dist", 0.60))
        self.yaw_gain = float(cfg.get("yaw_gain", 1.0))
        self.turn_slow_factor = float(cfg.get("turn_slow_factor", 0.25))

        # Command smoothing (0 = no smooth, 1 = fully smooth / never update)
        self.cmd_smoothing = float(cfg.get("cmd_smoothing", 0.70))

        # State
        self.step_counters = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.global_paths: list[torch.Tensor | None] = [None] * num_envs
        self.local_paths = torch.zeros(num_envs, self.local_path_length, 2, device=device)
        self.local_path_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.global_waypoint_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Command smoothing state
        self._prev_command = torch.zeros(num_envs, 3, device=device)

        # Metrics for rewards
        self.nearest_local_dist = torch.zeros(num_envs, device=device)
        self.on_local_path = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.path_progress = torch.zeros(num_envs, device=device)
        self.prev_path_progress = torch.zeros(num_envs, device=device)

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------
    def _height_scan_to_grid(self, height_scan: torch.Tensor):
        """height_scan (N,256) -> obstacle (N,16,16), True=obstacle."""
        grid = height_scan.view(self.num_envs, self.grid_h, self.grid_w)
        obstacle = grid < self.obstacle_threshold
        return obstacle, grid

    def _world_to_grid(self, pos_local: torch.Tensor) -> torch.Tensor:
        """
        pos_local: (N, 2) with (x=forward, y=left) in meters.
        Returns: (N, 2) int tensor of (row, col).
        """
        col = (pos_local[:, 0] / self.resolution).clamp(0, self.grid_w - 1).long()
        row = ((pos_local[:, 1] + 0.75) / self.resolution).clamp(0, self.grid_h - 1).long()
        return torch.stack([row, col], dim=1)

    def _grid_to_world(self, grid_pos: torch.Tensor) -> torch.Tensor:
        """
        grid_pos: (..., 2) with (row, col).
        Returns: (..., 2) with (x=forward, y=left) in meters.
        """
        world = torch.zeros_like(grid_pos, dtype=torch.float32)
        world[..., 0] = grid_pos[..., 1].float() * self.resolution
        world[..., 1] = (grid_pos[..., 0].float() - self.grid_h / 2 + 0.5) * self.resolution
        return world

    # ------------------------------------------------------------------
    # Global planning
    # ------------------------------------------------------------------
    def plan_global(self, height_scan: torch.Tensor, goal_local: torch.Tensor):
        """
        Plan global A* path for each env.
        Returns list of (N_waypoints, 2) tensors in grid coords.
        """
        obstacle, _ = self._height_scan_to_grid(height_scan)
        start = torch.full((self.num_envs, 2), self.grid_h // 2, dtype=torch.long, device=self.device)
        start[:, 1] = 0  # col = 0 (robot position)

        goal_grid = self._world_to_grid(goal_local)

        global_paths = []
        for i in range(self.num_envs):
            obs_grid = obstacle[i].cpu().numpy()
            # Inflate obstacles to keep path away from edges
            if self.inflation_radius > 0:
                obs_grid = inflate_obstacles(obs_grid, radius=self.inflation_radius)

            start_t = (start[i, 0].item(), start[i, 1].item())
            goal_t = (goal_grid[i, 0].item(), goal_grid[i, 1].item())

            # Quick check: is straight line blocked?
            if not line_blocked(obs_grid, start_t[0], start_t[1], goal_t[0], goal_t[1]):
                path = [start_t, goal_t]
            else:
                grid_free = ~obs_grid
                path = astar_single(grid_free, start_t, goal_t)
                if len(path) == 0:
                    path = [start_t, goal_t]

            # Subsample if too long
            if len(path) > self.global_path_max_length:
                indices = np.linspace(0, len(path) - 1, self.global_path_max_length).astype(int)
                path = [path[idx] for idx in indices]

            path_tensor = torch.tensor(path, dtype=torch.float32, device=self.device)
            global_paths.append(path_tensor)

        return global_paths

    # ------------------------------------------------------------------
    # Local planning
    # ------------------------------------------------------------------
    def plan_local(self, global_paths: list[torch.Tensor | None], height_scan: torch.Tensor):
        """
        Derive local path from global path + current obstacles.
        Returns (local_paths (N,L,2), active (N,)).
        """
        obstacle, _ = self._height_scan_to_grid(height_scan)
        start = (self.grid_h // 2, 0)

        local_paths = torch.zeros(self.num_envs, self.local_path_length, 2, device=self.device)
        active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for i in range(self.num_envs):
            gp = global_paths[i]
            if gp is None or gp.shape[0] == 0:
                continue

            # Take first waypoints from global path
            num_pts = min(int(gp.shape[0]), self.local_path_length)
            waypoints = gp[:num_pts].long()

            # Check if any waypoint is blocked (use non-inflated obstacle for local check)
            blocked = False
            first_blocked_idx = None
            for j in range(num_pts):
                r, c = waypoints[j, 0].item(), waypoints[j, 1].item()
                if 0 <= r < self.grid_h and 0 <= c < self.grid_w and obstacle[i, r, c]:
                    blocked = True
                    first_blocked_idx = j
                    break

            if not blocked:
                # Global path is clear — use directly
                for j in range(num_pts):
                    local_paths[i, j] = gp[j]
                for j in range(num_pts, self.local_path_length):
                    local_paths[i, j] = gp[num_pts - 1]
                active[i] = True
            else:
                # Need local replanning
                if first_blocked_idx is not None and first_blocked_idx > 0:
                    target_idx = min(first_blocked_idx + 3, int(gp.shape[0]) - 1)
                    target = gp[target_idx].long()

                    # Find nearest free cell around target
                    free_target = None
                    for radius in range(1, 5):
                        for dr in range(-radius, radius + 1):
                            for dc in range(-radius, radius + 1):
                                if abs(dr) != radius and abs(dc) != radius:
                                    continue
                                tr, tc = target[0].item() + dr, target[1].item() + dc
                                if 0 <= tr < self.grid_h and 0 <= tc < self.grid_w and not obstacle[i, tr, tc]:
                                    free_target = (tr, tc)
                                    break
                            if free_target:
                                break
                        if free_target:
                            break

                    if free_target:
                        obs_grid = obstacle[i].cpu().numpy()
                        if self.inflation_radius > 0:
                            obs_grid = inflate_obstacles(obs_grid, radius=self.inflation_radius)
                        path = astar_single(~obs_grid, start, free_target)
                        if len(path) > 0:
                            path_t = torch.tensor(path, dtype=torch.float32, device=self.device)
                            if path_t.shape[0] >= self.local_path_length:
                                local_paths[i] = path_t[:self.local_path_length]
                            else:
                                local_paths[i, :path_t.shape[0]] = path_t
                                local_paths[i, path_t.shape[0]:] = path_t[-1]
                            active[i] = True

        return local_paths, active

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------
    def update(self, obs: torch.Tensor):
        """
        Update global and local paths based on current observation.
        Call once per environment step.
        """
        self.step_counters += 1
        height_scan = obs[:, 45:301]

        # Extract goal local coordinates from obs (if available)
        goal_local = torch.zeros(self.num_envs, 2, device=self.device)
        if obs.shape[1] > 301:
            # goal_obs: [local_x, local_y, goal_dist] normalized to [-1,1] for XY, [0,1] for dist
            goal_obs = obs[:, 301:304]
            goal_local = goal_obs[:, :2] * 10.0  # denormalize to meters
        else:
            goal_local[:, 0] = 1.5  # default: straight ahead max range

        # Global planning (lower frequency, or once-only in static maze mode)
        if self.static_maze:
            # Only plan for envs that have no global path yet (e.g. after reset)
            need_global = torch.tensor(
                [self.global_paths[i] is None for i in range(self.num_envs)],
                device=self.device,
            )
        else:
            need_global = self.step_counters % self.global_plan_steps == 0

        if need_global.any():
            gps = self.plan_global(height_scan, goal_local)
            for i in range(self.num_envs):
                if need_global[i]:
                    self.global_paths[i] = gps[i]
                    self.global_waypoint_idx[i] = 0

        # Local planning (higher frequency)
        need_local = self.step_counters % self.local_plan_steps == 0
        if need_local.any():
            lp, act = self.plan_local(self.global_paths, height_scan)
            for i in range(self.num_envs):
                if need_local[i]:
                    self.local_paths[i] = lp[i]
                    self.local_path_active[i] = act[i]

        self._compute_path_metrics()

    # ------------------------------------------------------------------
    # Metrics for rewards
    # ------------------------------------------------------------------
    def _compute_path_metrics(self):
        """Compute distance to local path and progress metrics."""
        robot_pos = torch.zeros(self.num_envs, 2, device=self.device)
        path_world = self._grid_to_world(self.local_paths)  # (N, L, 2)

        # Distance to each waypoint
        dists = torch.norm(path_world - robot_pos.unsqueeze(1), dim=2)  # (N, L)
        self.nearest_local_dist, nearest_idx = dists.min(dim=1)
        self.on_local_path = self.nearest_local_dist < self.path_follow_tolerance

        # Progress: cumulative path length up to nearest waypoint
        self.prev_path_progress = self.path_progress.clone()
        # Vectorized progress computation
        seg_lengths = torch.norm(path_world[:, 1:] - path_world[:, :-1], dim=2)  # (N, L-1)
        idx_mask = torch.arange(self.local_path_length - 1, device=self.device).unsqueeze(0) < nearest_idx.unsqueeze(1)
        self.path_progress = (seg_lengths * idx_mask.float()).sum(dim=1)

    # ------------------------------------------------------------------
    # Navigation command generation
    # ------------------------------------------------------------------
    def get_navigation_command(self) -> torch.Tensor:
        """
        Generate velocity commands [vx, vy, yaw_rate] based on local path.
        Fully vectorized: no Python loops over envs or waypoints.
        Commands are temporally smoothed to avoid sudden jumps.
        Returns: (num_envs, 3) tensor.
        """
        raw_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        # Default for inactive envs: slow forward
        raw_cmd[:, 0] = 0.2

        if not self.local_path_active.any():
            alpha = max(0.0, min(1.0, self.cmd_smoothing))
            smoothed = alpha * raw_cmd + (1.0 - alpha) * self._prev_command
            self._prev_command = smoothed.clone()
            return smoothed

        # Vectorized: convert all local paths to world coords at once
        path_world = self._grid_to_world(self.local_paths)  # (N, L, 2)

        # Distance from robot (origin) to each waypoint
        dists = torch.norm(path_world, dim=2)  # (N, L)

        # Find first waypoint beyond min_target_dist for each env
        beyond_mask = (dists > self.min_target_dist) & self.local_path_active.unsqueeze(1)  # (N, L)
        first_beyond_idx = beyond_mask.int().argmax(dim=1)  # (N,)
        has_beyond = beyond_mask.any(dim=1)  # (N,)
        target_idx = torch.where(
            has_beyond,
            first_beyond_idx,
            torch.tensor(self.local_path_length - 1, device=self.device),
        )

        # Gather target waypoints: (N, 2)
        batch_idx = torch.arange(self.num_envs, device=self.device)
        targets = path_world[batch_idx, target_idx]

        # Compute target distances and angles
        target_dists = torch.norm(targets, dim=1)  # (N,)
        target_angles = torch.atan2(targets[:, 1], targets[:, 0].clamp(min=1e-6))  # (N,)

        # Forward speed
        forward_speed = target_dists.clamp(max=self.max_target_dist)
        turn_penalty = (1.0 - target_angles.abs() / (math.pi / 2)).clamp(min=0.0)
        forward_speed *= (self.turn_slow_factor + (1.0 - self.turn_slow_factor) * turn_penalty)

        # Yaw rate
        yaw_rate = (target_angles * self.yaw_gain).clamp(-1.0, 1.0)

        # Only override active envs (inactive keep the default 0.2 forward)
        raw_cmd[self.local_path_active, 0] = forward_speed[self.local_path_active]
        raw_cmd[self.local_path_active, 2] = yaw_rate[self.local_path_active]

        # Temporal smoothing
        alpha = max(0.0, min(1.0, self.cmd_smoothing))
        smoothed = alpha * raw_cmd + (1.0 - alpha) * self._prev_command
        self._prev_command = smoothed.clone()
        return smoothed

    # ------------------------------------------------------------------
    # Reward interfaces (called by reward_process)
    # ------------------------------------------------------------------
    def get_path_follow_reward(self) -> torch.Tensor:
        """
        High reward for staying close to local path.
        Gaussian falloff within tolerance. Sigma set to 0.8 * tolerance
        so the edge of tolerance still gets ~60% of max reward.
        """
        dist = self.nearest_local_dist
        sigma = self.path_follow_tolerance * 0.8
        reward = torch.exp(-dist ** 2 / (2 * sigma ** 2))
        return reward

    def get_path_progress_reward(self) -> torch.Tensor:
        """Reward for making progress along the local path."""
        progress = self.path_progress - self.prev_path_progress
        return progress.clamp(min=0.0)

    def get_stats(self) -> dict[str, float]:
        """Return aggregated stats for monitoring."""
        return {
            "nav_local_path_active": float(self.local_path_active.float().mean().item()),
            "nav_on_path_ratio": float(self.on_local_path.float().mean().item()),
            "nav_nearest_dist_mean": float(self.nearest_local_dist.mean().item()),
            "nav_path_progress_mean": float(self.path_progress.mean().item()),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_envs(self, env_ids: torch.Tensor):
        """Reset state for done environments."""
        if env_ids.numel() == 0:
            return
        self.step_counters[env_ids] = 0
        for idx in env_ids.tolist():
            self.global_paths[idx] = None
        self.local_path_active[env_ids] = False
        self.global_waypoint_idx[env_ids] = 0
        self.path_progress[env_ids] = 0.0
        self.prev_path_progress[env_ids] = 0.0
        self.nearest_local_dist[env_ids] = 0.0
        self.on_local_path[env_ids] = False
        self._prev_command[env_ids] = 0.0
        # In static maze mode, trigger global replanning immediately on reset
        # so the path is ready before the first step.
        if self.static_maze and env_ids.numel() > 0:
            # We need height_scan and goal_local to plan, but they are not
            # available here. The first call to update() will catch these envs
            # because their global_paths are None.
            pass
