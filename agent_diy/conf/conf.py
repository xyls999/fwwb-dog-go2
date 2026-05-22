#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Multi-stage training configuration for hybrid CPG + RL agent.
混合 CPG+RL 智能体的多阶段训练配置。

Stages:
  1. LocomotionConfig  — learn stable walking (standard terrain)
  2. NavConfig         — end-to-end navigation (track terrain)
  3. HierNavConfig     — hierarchical nav (frozen loco + trainable nav)

Switch stage by setting Config.CURRENT = <StageClass>.
"""

import os
import toml

_VALID_TASKS = {"standard", "track"}


class StageConfig:
    """Base training stage configuration."""

    name = ""
    task_type = "standard"

    # ─── Architecture dimensions (fixed by Isaac Lab) ───
    num_actions = 12
    num_proprio_obs = 45
    num_scan = 256
    num_critic_observations = 316
    num_goal_obs = 0  # 0 for standard, 4 for track

    # ─── Model architecture (lighter than baseline since CPG does heavy lifting) ───
    model_class = "ActorCritic"
    actor_hidden_dims = [256, 128]
    critic_hidden_dims = [256, 128]
    activation = "elu"
    residual_clip = 0.35

    # ─── Training hyperparameters ───
    lr = 3e-4
    num_learning_epochs = 5
    num_mini_batches = 4
    num_steps_per_env = 48
    min_normalized_std = [0.03, 0.015, 0.03] * 4  # smaller std since residuals are small

    # ─── Saving ───
    model_save_interval = 10

    # ─── CPG parameters ───
    cpg_base_freq = 2.8       # Hz, trot frequency
    cpg_amp_hip = 0.55        # hip oscillation amplitude
    cpg_amp_thigh = 0.70      # thigh lift amplitude
    cpg_amp_calf = 0.45       # calf flex amplitude
    cpg_prior_coef = 0.3      # L2 weight on CPG prior (anneals over training)

    # ─── Reflex parameters ───
    reflex_tip_threshold = 0.55
    reflex_ang_vel_threshold = 3.5
    reflex_recovery_duration = 15  # steps


class LocomotionConfig(StageConfig):
    """Stage 1: Learn stable locomotion on mixed standard terrain."""
    name = "locomotion"
    task_type = "standard"
    num_goal_obs = 0
    lr = 3e-4


class NavConfig(StageConfig):
    """Stage 2: End-to-end navigation on track terrain."""
    name = "nav"
    task_type = "track"
    num_goal_obs = 4
    lr = 1e-4  # lower LR for fine-tuning on nav
    num_steps_per_env = 48
    # Larger clip for nav exploration
    residual_clip = 0.40


class HierNavConfig(StageConfig):
    """Stage 3: Hierarchical nav — frozen loco + trainable nav policy."""
    name = "hier_nav"
    task_type = "track"
    num_goal_obs = 4
    lr = 1e-4
    # Nav policy only outputs velocity commands [vx, vy, ωz]
    nav_action_dim = 3


class Config:
    """Unified config entry. Set Config.CURRENT to switch stages."""

    CURRENT = LocomotionConfig

    @staticmethod
    def load_conf(logger):
        from common_python.config.config_control import CONFIG
        from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine

        stage = Config.CURRENT
        task_type = stage.task_type

        if task_type not in _VALID_TASKS:
            raise ValueError(
                f"Invalid task_type '{task_type}' in stage '{stage.name}'."
            )

        is_eval = False
        if hasattr(CONFIG, "run_mode"):
            is_eval = CONFIG.run_mode in [
                KaiwuDRLDefine.RUN_MODE_EVAL,
                KaiwuDRLDefine.RUN_MODE_EXAM,
            ]

        if is_eval:
            usr_conf_file = "tools/eval/conf/eval_env_conf.toml"
        else:
            usr_conf_file = f"agent_diy/conf/train_env_conf_{task_type}_{stage.name}.toml"

        usr_conf = _load_conf(usr_conf_file, logger)
        if usr_conf is None:
            raise Exception(f"usr_conf is None, please check {usr_conf_file}")

        logger.info(
            f"Stage: {stage.name}, task_type: {task_type}, "
            f"model: {stage.model_class}, obs_dim: {stage.num_proprio_obs + stage.num_scan + stage.num_goal_obs}"
        )
        return usr_conf, usr_conf_file, is_eval, stage


def _deep_merge(base, override):
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_conf(conf_file, logger):
    if not os.path.exists(conf_file):
        logger.error(f"Config file not found: {conf_file}")
        return None

    mode = "eval" if "eval" in conf_file else "train"
    base_file = os.path.join("tools", "conf", "base", f"{mode}_env_base.toml")

    base_config = {}
    if os.path.exists(base_file):
        try:
            with open(base_file, "r", encoding="utf-8") as f:
                base_config = toml.load(f)
            logger.info(f"Loaded base config: {base_file}")
        except Exception as e:
            logger.warning(f"Cannot load base config: {base_file}. Error: {e}")

    try:
        with open(conf_file, "r", encoding="utf-8") as f:
            user_config = toml.load(f)
        logger.info(f"Loaded user config: {conf_file}")
    except Exception as e:
        logger.error(f"Cannot load config file: {conf_file}. Error: {e}")
        return None

    return _deep_merge(base_config, user_config) if base_config else user_config
