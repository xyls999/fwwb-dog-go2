#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Stage and environment configuration entrypoint for agent_ppo.
"""


from __future__ import annotations

import os

try:
    import toml
except ModuleNotFoundError:
    toml = None
    import tomllib


VALID_TASK_TYPES = {"standard", "track"}


class WkStageConfig:
    """Base class for one PPO training stage."""

    name = ""
    task_type = "standard"

    # Note: Model architecture dimensions come from the Isaac Lab task definition and
    # Note: should stay in Python instead of user TOML so model shapes stay steady.
    num_actions = 12
    num_proprio_obs = 45
    num_scan = 256
    num_goal_obs = 0
    num_critic_observations = 316

    model_class = "WkActorCritic"
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"

    lr = 3e-4
    num_learning_epochs = 5
    num_mini_batches = 4
    num_steps_per_env = 48
    clip_param = 0.2
    entropy_coef = 0.01
    desired_kl = 0.01
    init_noise_std = 1.0
    min_normalized_std = [0.05, 0.02, 0.05] * 4
    max_normalized_std = [1.2, 0.8, 1.2] * 4

    model_save_interval = 50


class WkCustomStageConfig(WkStageConfig):
    """Template stage for future custom experiments."""


class WkLocomotionStageConfig(WkStageConfig):
    """Stable mixed-terrain locomotion pretraining."""

    name = "locomotion"
    task_type = "standard"


class WkStairConservativeStageConfig(WkStageConfig):
    """Conservative stair fine-tune while replaying simpler terrain."""

    name = "stair_conservative"
    task_type = "standard"
    lr = 1e-4
    num_learning_epochs = 3
    num_mini_batches = 4
    num_steps_per_env = 48
    model_save_interval = 50


class WkStairInvFineTuneStageConfig(WkStageConfig):
    """Fine-tune for higher and inverse stair variants."""

    name = "stair_inv_finetune"
    task_type = "standard"
    lr = 1e-4
    num_learning_epochs = 3
    num_mini_batches = 4
    num_steps_per_env = 48
    model_save_interval = 100


class WkTrackNavStageConfig(WkStageConfig):
    """Track-navigation fine-tune on top of a pretrained locomotion policy."""

    name = "nav"
    task_type = "track"
    num_goal_obs = 3
    num_critic_observations = 319
    lr = 1.2e-5
    num_learning_epochs = 3
    num_mini_batches = 4
    num_steps_per_env = 48
    entropy_coef = 0.0008
    desired_kl = 0.003
    init_noise_std = 0.80
    min_normalized_std = [0.05, 0.025, 0.05] * 4
    max_normalized_std = [0.24, 0.14, 0.24] * 4
    model_save_interval = 20


class WkRuntimeConfig:
    """Single configuration entrypoint used by the rest of agent_ppo."""

    CURRENT = WkTrackNavStageConfig

    @staticmethod
    def load_conf(logger):
        """Load the runtime env config for the currently selected stage."""

        from common_python.config.config_control import CONFIG
        from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine

        stage_config = WkRuntimeConfig.CURRENT
        task_type = stage_config.task_type

        if task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"Invalid task_type '{task_type}' in stage '{stage_config.name}'. "
                f"Only {VALID_TASK_TYPES} are supported."
            )

        is_eval_mode = False
        if hasattr(CONFIG, "run_mode"):
            is_eval_mode = CONFIG.run_mode in [
                KaiwuDRLDefine.RUN_MODE_EVAL,
                KaiwuDRLDefine.RUN_MODE_EXAM,
            ]

        if is_eval_mode:
            runtime_conf_path = "tools/eval/conf/eval_env_conf.toml"
        else:
            runtime_conf_path = f"agent_ppo/conf/train_env_conf_{task_type}_{stage_config.name}.toml"

        runtime_conf = wk_load_runtime_env_config(runtime_conf_path, logger)
        if runtime_conf is None:
            error_message = f"usr_conf is None, please check {runtime_conf_path}"
            logger.error(error_message)
            raise Exception(error_message)

        logger.info(
            f"Stage: {stage_config.name}, task_type: {task_type}, model: {stage_config.model_class}"
        )
        return runtime_conf, runtime_conf_path, is_eval_mode, stage_config


def wk_merge_dicts_recursively(base_dict, override_dict):
    """Recursively merge override_dict into base_dict."""

    merged_dict = base_dict.copy()
    for key, value in override_dict.items():
        if isinstance(value, dict) and isinstance(merged_dict.get(key), dict):
            merged_dict[key] = wk_merge_dicts_recursively(merged_dict[key], value)
        else:
            merged_dict[key] = value
    return merged_dict


def wk_load_toml_file(path):
    """Load TOML using `toml` when available, else Python's stdlib parser."""

    if toml is not None:
        with open(path, "r", encoding="utf-8") as file_obj:
            return toml.load(file_obj)
    with open(path, "rb") as file_obj:
        return tomllib.load(file_obj)


def wk_load_runtime_env_config(runtime_conf_path, logger):
    """Load base env config and overlay the stage-specific runtime config."""

    if not os.path.exists(runtime_conf_path):
        logger.error(f"Config file not found: {runtime_conf_path}")
        return None

    config_mode = "eval" if "eval" in runtime_conf_path else "train"
    base_conf_path = os.path.join("tools", "conf", "base", f"{config_mode}_env_base.toml")

    base_config = {}
    if os.path.exists(base_conf_path):
        try:
            base_config = wk_load_toml_file(base_conf_path)
            logger.info(f"Loaded base config: {base_conf_path}")
        except Exception as exc:
            logger.warning(f"Cannot load base config: {base_conf_path}. Error: {exc}")

    try:
        runtime_override_config = wk_load_toml_file(runtime_conf_path)
        logger.info(f"Loaded user config: {runtime_conf_path}")
    except Exception as exc:
        logger.error(f"Cannot load config file: {runtime_conf_path}. Error: {exc}")
        return None

    if base_config:
        return wk_merge_dicts_recursively(base_config, runtime_override_config)
    return runtime_override_config


# Note: Compatibility aliases for platform-side or older local imports.  The runtime
# Note: code above uses Wk* names, while these aliases keep the public surface steady.
Config = WkRuntimeConfig
StageConfig = WkStageConfig
CustomConfig = WkCustomStageConfig
LocomotionConfig = WkLocomotionStageConfig
StairConservativeConfig = WkStairConservativeStageConfig
StairInvFineTuneConfig = WkStairInvFineTuneStageConfig
TrackNavConfig = WkTrackNavStageConfig
merge_dicts_recursively = wk_merge_dicts_recursively
load_toml_file = wk_load_toml_file
load_runtime_env_config = wk_load_runtime_env_config
_VALID_TASKS = VALID_TASK_TYPES
_deep_merge = wk_merge_dicts_recursively
_load_toml = wk_load_toml_file
_load_conf = wk_load_runtime_env_config
