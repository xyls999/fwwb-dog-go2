# -*- coding: UTF-8 -*-
"""Track-aware goal helpers shared by observation and reward code."""

import torch

from agent_ppo.conf.conf import Config


def _env_track_length(env):
    try:
        terrain = env.scene.terrain
        terrain_gen_cfg = getattr(terrain.cfg, "terrain_generator", None)
        track_length = getattr(terrain_gen_cfg, "track_length", None)
        if track_length is not None:
            return int(track_length)
    except Exception:
        pass
    return None


def _track_settings(
    env=None,
    track_total_length: float | None = None,
    track_num_segments: int | None = None,
    track_maze_segment: int | None = None,
):
    stage = Config.CURRENT
    env_num_segments = _env_track_length(env) if track_num_segments is None and env is not None else None
    cfg_segments = int(getattr(stage, "track_num_segments", 5))
    num_segments = int(track_num_segments if track_num_segments is not None else (env_num_segments or cfg_segments))
    num_segments = max(num_segments, 1)
    if track_total_length is not None:
        total_length = float(track_total_length)
    elif env_num_segments is not None and env_num_segments != cfg_segments:
        total_length = float(getattr(stage, "track_segment_length", 7.2)) * float(num_segments)
    else:
        total_length = float(getattr(stage, "track_total_length", 36.0))
    maze_segment = int(
        track_maze_segment
        if track_maze_segment is not None
        else getattr(stage, "track_maze_segment", num_segments - 1)
    )
    maze_segment = max(0, min(maze_segment, num_segments - 1))
    return max(total_length, 1.0), num_segments, maze_segment


def _robot_root(env):
    try:
        robot = env.scene["robot"]
        return robot.data.root_pos_w, robot.data.root_quat_w
    except Exception:
        return None, None


def _final_goal_xy(env):
    if not hasattr(env, "goal_positions") or env.goal_positions is None:
        return None
    return env.goal_positions[:, :2]


def track_phase(
    env,
    track_total_length: float | None = None,
    track_num_segments: int | None = None,
    track_maze_segment: int | None = None,
):
    """Estimate full-track progress from x position and the final goal."""
    root_pos_w, _ = _robot_root(env)
    final_goal = _final_goal_xy(env)
    device = getattr(env, "device", root_pos_w.device if root_pos_w is not None else "cpu")
    num_envs = getattr(env, "num_envs", final_goal.shape[0] if final_goal is not None else 1)

    if root_pos_w is None or final_goal is None:
        zeros = torch.zeros(num_envs, device=device)
        return {
            "progress": zeros,
            "segment": torch.zeros(num_envs, dtype=torch.long, device=device),
            "pre_maze_gate": zeros,
            "maze_gate": zeros,
            "start_x": zeros,
            "final_x": zeros,
        }

    total_length, num_segments, maze_segment = _track_settings(
        env,
        track_total_length, track_num_segments, track_maze_segment
    )
    final_x = final_goal[:, 0]
    start_x = final_x - total_length
    progress = torch.clamp((root_pos_w[:, 0] - start_x) / total_length, 0.0, 1.0)
    segment = torch.floor(progress * float(num_segments)).long()
    segment = torch.clamp(segment, 0, num_segments - 1)
    pre_maze_gate = (segment < maze_segment).float()
    maze_gate = (segment >= maze_segment).float()
    return {
        "progress": progress,
        "segment": segment,
        "pre_maze_gate": pre_maze_gate,
        "maze_gate": maze_gate,
        "start_x": start_x,
        "final_x": final_x,
    }


def dynamic_track_goal_xy(
    env,
    use_subgoal: bool | None = None,
    track_total_length: float | None = None,
    track_num_segments: int | None = None,
    track_maze_segment: int | None = None,
):
    """Return current subgoal xy.

    Before the maze there is no meaningful per-segment exit for the policy to
    chase.  The useful behavior is a side-lane runner: keep a safe lateral
    offset and move forward.  Once the robot reaches the maze segment, the
    target switches to the official final goal.
    """
    final_goal = _final_goal_xy(env)
    if final_goal is None:
        return None, track_phase(env, track_total_length, track_num_segments, track_maze_segment)

    stage = Config.CURRENT
    if use_subgoal is None:
        use_subgoal = bool(getattr(stage, "use_track_subgoals", True))
    phase = track_phase(env, track_total_length, track_num_segments, track_maze_segment)
    if not use_subgoal:
        return final_goal, phase

    total_length, num_segments, maze_segment = _track_settings(
        env,
        track_total_length, track_num_segments, track_maze_segment
    )
    root_pos_w, _ = _robot_root(env)
    segment_len = total_length / float(num_segments)
    maze_entry_x = phase["start_x"] + float(maze_segment) * segment_len
    lookahead = float(getattr(stage, "track_forward_lookahead", 3.0))
    if root_pos_w is None:
        forward_x = maze_entry_x
    else:
        forward_x = root_pos_w[:, 0] + lookahead
        if bool(getattr(stage, "track_cap_forward_target_to_maze_entry", True)):
            # A tiny overshoot avoids the target falling behind right at the
            # boundary, but this is still a forward lane target, not a segment
            # exit target.
            forward_x = torch.minimum(forward_x, maze_entry_x + 0.6)
    side_offset_y = float(getattr(stage, "track_side_offset_y", 0.0))
    target_x = torch.where(phase["segment"] < maze_segment, forward_x, final_goal[:, 0])
    target_y = torch.where(
        phase["segment"] < maze_segment,
        final_goal[:, 1] + side_offset_y,
        final_goal[:, 1],
    )
    return torch.stack((target_x, target_y), dim=1), phase


def goal_delta_body(
    env,
    use_subgoal: bool | None = None,
    track_total_length: float | None = None,
    track_num_segments: int | None = None,
    track_maze_segment: int | None = None,
):
    root_pos_w, quat = _robot_root(env)
    target_xy, _ = dynamic_track_goal_xy(
        env,
        use_subgoal=use_subgoal,
        track_total_length=track_total_length,
        track_num_segments=track_num_segments,
        track_maze_segment=track_maze_segment,
    )
    device = getattr(env, "device", root_pos_w.device if root_pos_w is not None else "cpu")
    num_envs = getattr(env, "num_envs", target_xy.shape[0] if target_xy is not None else 1)
    if root_pos_w is None or quat is None or target_xy is None:
        return (
            torch.zeros(num_envs, 2, device=device),
            torch.zeros(num_envs, device=device),
        )

    delta_w = target_xy - root_pos_w[:, :2]
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    heading = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cos_h = torch.cos(-heading)
    sin_h = torch.sin(-heading)
    local_x = cos_h * delta_w[:, 0] - sin_h * delta_w[:, 1]
    local_y = sin_h * delta_w[:, 0] + cos_h * delta_w[:, 1]
    local = torch.stack((local_x, local_y), dim=1)
    return local, torch.linalg.norm(delta_w, dim=1)


def goal_observation_features(env, feature_dim: int):
    if feature_dim <= 0:
        return None
    zeros = torch.zeros(env.num_envs, feature_dim, device=env.device)

    local_goal, subgoal_dist = goal_delta_body(env, use_subgoal=True)
    _, final_dist = goal_delta_body(env, use_subgoal=False)
    goal_scale = float(getattr(Config.CURRENT, "goal_feature_scale", 10.0))
    dist_scale = float(getattr(Config.CURRENT, "goal_distance_feature_scale", 20.0))
    local_goal_scaled = torch.clamp(local_goal / max(goal_scale, 1e-6), -1.0, 1.0)
    final_dist_scaled = torch.clamp(final_dist, 0.0, dist_scale) / max(dist_scale, 1e-6)
    phase = track_phase(env)

    if feature_dim == 3:
        return torch.cat((local_goal_scaled, final_dist_scaled.unsqueeze(1)), dim=1)

    if feature_dim == 5:
        denom = torch.clamp(subgoal_dist, min=1e-6).unsqueeze(1)
        goal_dir = local_goal / denom
        forward_dir = torch.zeros_like(goal_dir)
        forward_dir[:, 0] = 1.0
        goal_dir = torch.where(phase["pre_maze_gate"].unsqueeze(1) > 0.5, forward_dir, goal_dir)
        return torch.cat((local_goal_scaled, goal_dir, final_dist_scaled.unsqueeze(1)), dim=1)

    return zeros


def goal_velocity_command(
    env,
    forward_min: float = 0.12,
    forward_max: float = 1.0,
    yaw_gain: float = 1.8,
    yaw_limit: float = 1.4,
    turn_slow_angle: float = 0.75,
    turn_in_place_angle: float = 1.15,
    turn_in_place_speed: float = 0.10,
    stop_distance: float = 0.45,
    slow_distance: float = 1.4,
    use_subgoal: bool | None = True,
):
    """Build a target-facing base velocity command from the dynamic track goal."""
    local_goal, dist = goal_delta_body(env, use_subgoal=use_subgoal)
    angle = torch.atan2(local_goal[:, 1], local_goal[:, 0])
    if use_subgoal is not False:
        phase = track_phase(env)
        angle = torch.where(phase["pre_maze_gate"] > 0.5, torch.zeros_like(angle), angle)
    abs_angle = torch.abs(angle)
    align = torch.clamp(torch.cos(angle), min=0.0, max=1.0)
    turn_scale = torch.clamp(1.0 - abs_angle / max(turn_slow_angle, 1e-6), min=0.0, max=1.0)
    speed = forward_min + (forward_max - forward_min) * torch.square(align) * turn_scale
    speed = torch.where(
        abs_angle > turn_in_place_angle,
        torch.full_like(speed, turn_in_place_speed),
        speed,
    )
    distance_gate = torch.clamp(
        (dist - stop_distance) / max(slow_distance - stop_distance, 1e-6),
        min=0.0,
        max=1.0,
    )
    speed = speed * distance_gate
    yaw = torch.clamp(yaw_gain * angle, min=-yaw_limit, max=yaw_limit)
    return torch.stack((speed, torch.zeros_like(speed), yaw), dim=1)
