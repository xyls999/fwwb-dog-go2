#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.utils.train_test_utils import run_train_test

# Note: To run train_test, apply the PPO agent configured in conf/algo_conf_legged_robot_competition_26.toml.
# Note: Simply modify the value of the algorithm_name variable.
# 说明：本工程维持 PPO 主线，尽量防止读取已经移除的实验 agent。
algorithm_name_list = ["ppo"]
algorithm_name = "ppo"


if __name__ == "__main__":
    run_train_test(
        algorithm_name=algorithm_name,
        algorithm_name_list=algorithm_name_list,
        env_vars={
            "replay_buffer_capacity": "10",
            "preload_ratio": "10",
            "train_batch_size": "2",
            "dump_model_freq": "1",
            "max_frame_no": "1000",
        },
        shell="bash",
        skip_aisrv_alive_check=True,
        skip_error_scan=True,
        check_model_method="listdir_count",
    )
