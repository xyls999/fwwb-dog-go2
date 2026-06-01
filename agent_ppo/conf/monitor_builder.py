#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Monitor panel definitions for PPO training.
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """Build the dashboard layout used by PPO training."""

    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("Quad Robot Nav")
        .add_group(group_name="Train Progress", group_name_en="training_progress")
        .add_panel(name="Mean Ep Length", name_en="mean_episode_length", type="line")
            .add_metric(metrics_name="mean_episode_length", expr="avg(mean_episode_length{})")
            .end_panel()
        .add_panel(name="Mean Ep Reward", name_en="mean_episode_reward", type="line")
            .add_metric(metrics_name="mean_episode_reward", expr="avg(mean_episode_reward{})")
            .end_panel()
        .end_group()
        .add_group(group_name="Algorithm", group_name_en="algorithm")
        .add_panel(name="Total Loss", name_en="total_loss", type="line")
            .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
            .end_panel()
        .add_panel(name="Value Loss", name_en="value_loss", type="line")
            .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
            .end_panel()
        .add_panel(name="Policy Loss", name_en="policy_loss", type="line")
            .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
            .end_panel()
        .add_panel(name="Entropy Loss", name_en="entropy_loss", type="line")
            .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
            .end_panel()
        .end_group()
        .add_group(group_name="Vel Tracking", group_name_en="velocity_tracking")
        .add_panel(name="Lin Vel Track", name_en="reward_track_lin_vel_xy", type="line")
            .add_metric(metrics_name="reward_track_lin_vel_xy", expr="avg(reward_track_lin_vel_xy{})")
            .end_panel()
        .add_panel(name="Yaw Vel Track", name_en="reward_track_ang_vel_z", type="line")
            .add_metric(metrics_name="reward_track_ang_vel_z", expr="avg(reward_track_ang_vel_z{})")
            .end_panel()
        .add_panel(name="Curr Track Ratio", name_en="vel_curriculum_tracking_ratio", type="line")
            .add_metric(
                metrics_name="vel_curriculum_tracking_ratio",
                expr="avg(vel_curriculum_tracking_ratio{})",
            )
            .end_panel()
        .add_panel(name="Curr Stage", name_en="vel_curriculum_stage", type="line")
            .add_metric(metrics_name="vel_curriculum_stage", expr="avg(vel_curriculum_stage{})")
            .end_panel()
        .end_group()
        .add_group(group_name="Posture", group_name_en="posture_quality")
        .add_panel(name="Flat Orient Pen", name_en="reward_flat_orientation", type="line")
            .add_metric(metrics_name="reward_flat_orientation", expr="avg(reward_flat_orientation{})")
            .end_panel()
        .add_panel(name="Base Height Pen", name_en="reward_correct_base_height", type="line")
            .add_metric(metrics_name="reward_correct_base_height", expr="avg(reward_correct_base_height{})")
            .end_panel()
        .add_panel(
            name="Joint Pos Pen",
            name_en="reward_joint_position_penalty",
            type="line",
        )
            .add_metric(
                metrics_name="reward_joint_position_penalty",
                expr="avg(reward_joint_position_penalty{})",
            )
            .end_panel()
        .add_panel(name="Lat Drift Pen", name_en="reward_base_lateral_vel", type="line")
            .add_metric(metrics_name="reward_base_lateral_vel", expr="avg(reward_base_lateral_vel{})")
            .end_panel()
        .add_panel(name="Pitch Roll Rate", name_en="reward_ang_vel_xy", type="line")
            .add_metric(metrics_name="reward_ang_vel_xy", expr="avg(reward_ang_vel_xy{})")
            .end_panel()
        .end_group()
        .add_group(group_name="Gait", group_name_en="gait_quality")
        .add_panel(name="Feet Air Time", name_en="reward_feet_air_time", type="line")
            .add_metric(metrics_name="reward_feet_air_time", expr="avg(reward_feet_air_time{})")
            .end_panel()
        .add_panel(name="Joint Vel Pen", name_en="reward_dof_vel", type="line")
            .add_metric(metrics_name="reward_dof_vel", expr="avg(reward_dof_vel{})")
            .end_panel()
        .add_panel(
            name="AirTime Var Pen",
            name_en="reward_air_time_variance_penalty",
            type="line",
        )
            .add_metric(
                metrics_name="reward_air_time_variance_penalty",
                expr="avg(reward_air_time_variance_penalty{})",
            )
            .end_panel()
        .add_panel(name="Pivot Turn Pen", name_en="reward_pivot_turning", type="line")
            .add_metric(metrics_name="reward_pivot_turning", expr="avg(reward_pivot_turning{})")
            .end_panel()
        .add_panel(name="Feet Slide Pen", name_en="reward_feet_slide", type="line")
            .add_metric(metrics_name="reward_feet_slide", expr="avg(reward_feet_slide{})")
            .end_panel()
        .add_panel(name="Feet Stumble Pen", name_en="reward_feet_stumble", type="line")
            .add_metric(metrics_name="reward_feet_stumble", expr="avg(reward_feet_stumble{})")
            .end_panel()
        .end_group()
        .add_group(group_name="Stab Contact", group_name_en="stability_contact")
        .add_panel(name="Vert Vel Pen", name_en="reward_lin_vel_z", type="line")
            .add_metric(metrics_name="reward_lin_vel_z", expr="avg(reward_lin_vel_z{})")
            .end_panel()
        .add_panel(name="Bad Contact Pen", name_en="reward_undesired_contacts", type="line")
            .add_metric(metrics_name="reward_undesired_contacts", expr="avg(reward_undesired_contacts{})")
            .end_panel()
        .add_panel(name="Termination Penalty", name_en="reward_termination", type="line")
            .add_metric(metrics_name="reward_termination", expr="avg(reward_termination{})")
            .end_panel()
        .add_panel(name="Joint Limit Pen", name_en="reward_dof_pos_limits", type="line")
            .add_metric(metrics_name="reward_dof_pos_limits", expr="avg(reward_dof_pos_limits{})")
            .end_panel()
        .end_group()
        .add_group(
            group_name="Action Smooth",
            group_name_en="joint_action_smoothness",
        )
        .add_panel(name="Joint Acc Pen", name_en="reward_joint_acc", type="line")
            .add_metric(metrics_name="reward_joint_acc", expr="avg(reward_joint_acc{})")
            .end_panel()
        .add_panel(name="Action Rate Penalty", name_en="reward_action_rate", type="line")
            .add_metric(metrics_name="reward_action_rate", expr="avg(reward_action_rate{})")
            .end_panel()
        .add_panel(name="Action Smooth Pen", name_en="reward_action_smoothness", type="line")
            .add_metric(metrics_name="reward_action_smoothness", expr="avg(reward_action_smoothness{})")
            .end_panel()
        .end_group()
        .add_group(group_name="Energy Torque", group_name_en="energy_torque")
        .add_panel(name="Energy Penalty", name_en="reward_energy", type="line")
            .add_metric(metrics_name="reward_energy", expr="avg(reward_energy{})")
            .end_panel()
        .add_panel(name="Joint Torque Pen", name_en="reward_joint_torques", type="line")
            .add_metric(metrics_name="reward_joint_torques", expr="avg(reward_joint_torques{})")
            .end_panel()
        .end_group()
        .add_group(group_name="RL Nav", group_name_en="rl_navigation")
        .add_panel(name="Fwd Head Vel", name_en="reward_forward_heading_velocity", type="line")
            .add_metric(
                metrics_name="reward_forward_heading_velocity",
                expr="avg(reward_forward_heading_velocity{})",
            )
            .end_panel()
        .add_panel(name="Backward Penalty", name_en="reward_backward_penalty", type="line")
            .add_metric(metrics_name="reward_backward_penalty", expr="avg(reward_backward_penalty{})")
            .end_panel()
        .add_panel(name="Goal Head Align", name_en="reward_goal_heading_alignment", type="line")
            .add_metric(
                metrics_name="reward_goal_heading_alignment",
                expr="avg(reward_goal_heading_alignment{})",
            )
            .end_panel()
        .add_panel(
            name="Goal Vel Proj",
            name_en="reward_goal_velocity_projection",
            type="line",
        )
            .add_metric(
                metrics_name="reward_goal_velocity_projection",
                expr="avg(reward_goal_velocity_projection{})",
            )
            .end_panel()
        .add_panel(
            name="Goal Back Pen",
            name_en="reward_goal_backtrack_penalty",
            type="line",
        )
            .add_metric(
                metrics_name="reward_goal_backtrack_penalty",
                expr="avg(reward_goal_backtrack_penalty{})",
            )
            .end_panel()
        .add_panel(name="Approach Goal", name_en="reward_approach_goal", type="line")
            .add_metric(metrics_name="reward_approach_goal", expr="avg(reward_approach_goal{})")
            .end_panel()
        .add_panel(name="Goal Distance", name_en="reward_goal_distance", type="line")
            .add_metric(metrics_name="reward_goal_distance", expr="avg(reward_goal_distance{})")
            .end_panel()
        .add_panel(name="Reach Goal", name_en="reward_reach_goal", type="line")
            .add_metric(metrics_name="reward_reach_goal", expr="avg(reward_reach_goal{})")
            .end_panel()
        .add_panel(name="Task Complete", name_en="reward_task_complete", type="line")
            .add_metric(metrics_name="reward_task_complete", expr="avg(reward_task_complete{})")
            .end_panel()
        .add_panel(name="Nav Time", name_en="reward_navigation_time", type="line")
            .add_metric(metrics_name="reward_navigation_time", expr="avg(reward_navigation_time{})")
            .end_panel()
        .add_panel(name="Maze Ctx Gate", name_en="reward_maze_context_gate", type="line")
            .add_metric(metrics_name="reward_maze_context_gate", expr="avg(reward_maze_context_gate{})")
            .end_panel()
        .add_panel(name="Wall Coll Pen", name_en="reward_wall_collision", type="line")
            .add_metric(metrics_name="reward_wall_collision", expr="avg(reward_wall_collision{})")
            .end_panel()
        .add_panel(name="Wall Stall Pen", name_en="reward_wall_stall_penalty", type="line")
            .add_metric(metrics_name="reward_wall_stall_penalty", expr="avg(reward_wall_stall_penalty{})")
            .end_panel()
        .add_panel(name="Wall Prox Pen", name_en="reward_wall_proximity", type="line")
            .add_metric(metrics_name="reward_wall_proximity", expr="avg(reward_wall_proximity{})")
            .end_panel()
        .add_panel(name="Open Space Reward", name_en="reward_open_space", type="line")
            .add_metric(metrics_name="reward_open_space", expr="avg(reward_open_space{})")
            .end_panel()
        .add_panel(name="Corridor Centering", name_en="reward_corridor_centering", type="line")
            .add_metric(metrics_name="reward_corridor_centering", expr="avg(reward_corridor_centering{})")
            .end_panel()
        .add_panel(name="Dir Explore", name_en="reward_directed_exploration", type="line")
            .add_metric(
                metrics_name="reward_directed_exploration",
                expr="avg(reward_directed_exploration{})",
            )
            .end_panel()
        .add_panel(name="Stuck Penalty", name_en="reward_stuck_penalty", type="line")
            .add_metric(metrics_name="reward_stuck_penalty", expr="avg(reward_stuck_penalty{})")
            .end_panel()
        .end_group()
        .add_group(group_name="Physics Obs", group_name_en="physics_obs")
        .add_panel(name="Fwd Vel Ref", name_en="velocity_x_stat", type="stat")
            .add_metric(metrics_name="cmd_vx", expr="avg(obs_cmd_vel_x{})")
            .add_metric(metrics_name="actual_vx", expr="avg(obs_actual_vel_x{})")
            .end_panel()
        .add_panel(name="Lat Vel Ref", name_en="velocity_y_stat", type="stat")
            .add_metric(metrics_name="cmd_vy", expr="avg(obs_cmd_vel_y{})")
            .add_metric(metrics_name="actual_vy", expr="avg(obs_actual_vel_y{})")
            .end_panel()
        .add_panel(name="Yaw Vel Ref", name_en="velocity_yaw_stat", type="stat")
            .add_metric(metrics_name="cmd_yaw", expr="avg(obs_cmd_yaw{})")
            .add_metric(metrics_name="actual_yaw", expr="avg(obs_actual_yaw{})")
            .end_panel()
        .add_panel(name="Vel Err Ref", name_en="velocity_error_stat", type="stat")
            .add_metric(metrics_name="vx_error", expr="avg(obs_lin_vel_x_error{})")
            .add_metric(metrics_name="yaw_error", expr="avg(obs_yaw_error{})")
            .end_panel()
        .add_panel(name="Fwd Vel Err", name_en="obs_lin_vel_x_error", type="line")
            .add_metric(metrics_name="obs_lin_vel_x_error", expr="avg(obs_lin_vel_x_error{})")
            .end_panel()
        .add_panel(name="Lat Vel Err", name_en="obs_lin_vel_y_error", type="line")
            .add_metric(metrics_name="obs_lin_vel_y_error", expr="avg(obs_lin_vel_y_error{})")
            .end_panel()
        .add_panel(name="Actual Fwd Vel", name_en="obs_actual_vel_x", type="line")
            .add_metric(metrics_name="obs_actual_vel_x", expr="avg(obs_actual_vel_x{})")
            .end_panel()
        .add_panel(name="Base Height", name_en="obs_base_height", type="line")
            .add_metric(metrics_name="obs_base_height", expr="avg(obs_base_height{})")
            .end_panel()
        .add_panel(name="Pitch Roll Rate", name_en="obs_ang_vel_xy", type="line")
            .add_metric(metrics_name="obs_ang_vel_xy", expr="avg(obs_ang_vel_xy{})")
            .end_panel()
        .end_group()
        .build()
    )
    return config_dict
