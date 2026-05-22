# 奖励函数配置与评分标准对齐报告 1

## 1. 同步状态

已将 PPO Track 相关配置同步到 `code-legged_robot_competition_26-IDE-22.0.12 (2).zip` 版本，且同步后下列文件与 zip 完全一致：

- `agent_ppo/conf/conf.py`
- `agent_ppo/conf/train_env_conf_track_nav.toml`
- `agent_ppo/feature/reward_process.py`
- `agent_ppo/workflow/train_workflow.py`
- `agent_ppo/conf/monitor_builder.py`

同步后已通过：

- `agent_ppo/conf/train_env_conf_track_nav.toml` TOML 解析
- `agent_ppo/conf/conf.py`
- `agent_ppo/feature/reward_process.py`
- `agent_ppo/workflow/train_workflow.py`
- `agent_ppo/conf/monitor_builder.py`

## 2. 秘密武器评分标准摘要

Track 最终分数：

```text
total_score = completion_coeff * (0.4 * time_score + 0.4 * pose_score + 0.2 * energy_score)
```

关键含义：

- `completion_coeff` 只由是否 `goal_reached` 决定。未到终点时，总分直接为 0。
- `pose_score = 100 * exp(-5 * mean_deviation)`，其中 `mean_deviation = episode 平均(|roll| + |pitch|)`。
- `energy_score = 100 * exp(-0.01 * mean_energy)`，其中 `mean_energy = episode 平均(sum(abs(joint_vel) * abs(applied_torque)))`。
- `time_score` 只有完成后才有意义，因为 Track 总分外面乘 `completion_coeff`。

直接结论：

- 第一优先级：完成 Track，保证 `completion_coeff`。
- 第二优先级：时间和姿态同权重，都是 0.4。
- 第三优先级：能耗权重 0.2，但如果能耗很低，会明显拖总分。

## 3. 当前 PPO Track 超参数

来源：`agent_ppo/conf/conf.py::TrackNavConfig`

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `lr` | `1.5e-5` | Track fine-tune 学习率，比较保守 |
| `num_learning_epochs` | `3` | 每批 rollout 训练轮数 |
| `num_mini_batches` | `4` | PPO minibatch 数量 |
| `num_steps_per_env` | `48` | 每环境 rollout 步数 |
| `entropy_coef` | `0.0008` | 探索强度，偏低，减少动作随机抖动 |
| `desired_kl` | `0.003` | KL 自适应学习率目标，偏保守 |
| `init_noise_std` | `0.8` | 初始动作分布标准差 |
| `min_normalized_std` | `[0.05, 0.025, 0.05] * 4` | 每类关节最小动作 std |
| `max_normalized_std` | `[0.24, 0.14, 0.24] * 4` | 每类关节最大动作 std，限制探索和高频动作 |
| `num_goal_obs` | `3` | policy obs 追加目标方向/距离 |
| `num_critic_observations` | `319` | critic obs = 316 + 3 |

对齐判断：

- 这套 PPO 超参偏稳定、低探索，适合 fine-tune 已经能完成的策略。
- 低 `entropy_coef` 和低 `max_normalized_std` 有利于减少高频动作，从而间接改善 `energy_score`。

## 4. 地图和速度参数

### 4.1 地图采样

```toml
[terrain.level_mix]
levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
weights = [0.02, 0.03, 0.045, 0.065, 0.085, 0.115, 0.17, 0.18, 0.18, 0.11]
pool_size = 100
```

对齐判断：

- `l6~l9` 合计 `64%`，明显偏高难度。
- `l7/l8` 最高，各 `18%`，适合针对中高难 Track 提升完成率和时间。
- `l9` 是 `11%`，没有最高，说明它不是一味堆最难，而是重点压在可学、可稳定提升的高难区间。

### 4.2 分阶段速度命令

```toml
pre_maze_lin_vel_x = [1.12, 1.34]
slope_lin_vel_x = [1.02, 1.24]
stairs_lin_vel_x = [0.76, 0.94]
maze_lin_vel_x = [1.16, 1.38]
eval_command = [1.18, 0.0, 0.0]
phase_command_resample_steps = 160
```

对齐判断：

- 速度目标明显偏高，主要服务 `time_score`。
- 楼梯段比其他阶段慢，减少姿态崩和撞台阶。
- 迷宫速度也高，依赖 `maze_anticipatory_turn` 和墙体惩罚兜底。
- 风险是 `energy_score` 被高速度、高关节速度拉低。

## 5. 当前完整 Reward 配置

### 5.1 速度与时间

| reward | weight | params | 对齐目标 |
|---|---:|---|---|
| `track_lin_vel_xy` | `2.35` | `std=0.32`, `command_name=base_velocity` | time，保持高速步态 |
| `command_speed_advantage` | `0.6` | `deadband=0.02`, `surplus_scale=0.32`, `lag_scale=0.3`, `max_surplus=0.24`, `max_lag=0.6`, `lag_penalty_scale=0.95`, `min_command=0.1` | time，奖励略快于命令 |
| `track_ang_vel_z` | `0.12` | `std=0.35`, `command_name=base_velocity` | 姿态/导航，弱 yaw 跟踪 |
| `forward_heading_velocity` | `1.85` | `target_speed=1.25`, `max_reward=1.0` | time，身体朝前高速移动 |
| `goal_velocity_projection` | `4.6` | `max_speed=1.45` | time/completion，朝目标方向推进 |
| `navigation_time` | `-0.022` | 无 | time，每步扣分，鼓励快到终点 |

对齐判断：

- 这组是当前配置里最强的时间分驱动。
- `goal_velocity_projection=4.6` 和 `navigation_time=-0.022` 会强迫策略更快完成。
- 如果能量分低，优先怀疑这组和高速度命令共同导致关节功率上升。

### 5.2 完成率与目标到达

| reward | weight | params | 对齐目标 |
|---|---:|---|---|
| `approach_goal` | `13.0` | 无 | completion，缩短到终点距离 |
| `goal_distance` | `0.8` | `scale=8.0` | completion，距离越近越好 |
| `reach_goal` | `30.0` | `threshold=0.6` | completion，到达终点半径 |
| `task_complete` | `155.0` | `threshold=0.6` | completion_coeff，强稀疏完成奖励 |
| `goal_heading_alignment` | `1.0` | `std=0.75` | completion/time，朝向目标 |
| `goal_backtrack_penalty` | `-3.0` | `deadband=0.02` | completion，惩罚远离目标 |
| `backward_penalty` | `-2.5` | `deadband=0.03` | completion/time，防止倒车 |
| `directed_exploration` | `0.035` | `radius=0.55`, `memory_size=96`, `goal_heading_std=1.0` | completion，迷宫探索 |
| `stuck_penalty` | `-1.8` | `min_command=0.05`, `still_speed=0.12` | completion/time，防止卡住 |

对齐判断：

- `task_complete=155` 是最大单项正奖励，明确对齐 `completion_coeff`。
- `threshold=0.6` 与平台 goal radius 对齐，避免终止和奖励死区。
- completion 驱动足够强，适合 Track，因为没完成总分直接归 0。

### 5.3 姿态分

| reward | weight | params | 对齐目标 |
|---|---:|---|---|
| `pose_score_formula` | `1.0` | 无 | pose，近似 `exp(-5*(|roll|+|pitch|))` |
| `flat_orientation` | `-1.25` | 无 | pose，抑制机身倾斜 |
| `ang_vel_xy` | `-0.48` | 无 | pose，抑制 roll/pitch 角速度 |
| `lin_vel_z` | `-0.52` | 无 | pose/energy，减少跳跃 |
| `correct_base_height` | `-0.2` | `target_height=0.38` | pose，保持机身高度 |
| `rough_ang_vel_xy` | `-0.05` | rough gate: `body_y=4..12`, `near_x=1..10`, `delta_quantile=0.85`, `min_delta=0.025`, `full_delta=0.1` | rough terrain pose |
| `rough_roll_pitch_abs` | `-0.145` | rough gate: `min_delta=0.015`, `full_delta=0.075` | rough terrain pose |
| `rough_score_guidance` | `0.48` | `tracking_std=0.9`, `posture_weight=0.38`, rough gate | rough terrain pose/energy |

对齐判断：

- `pose_score_formula` 直接对齐平台姿态指数公式。
- `rough_*` 只在坡/楼梯等粗糙地形增强姿态约束，避免全程过度保守。
- 姿态权重中等，不如时间/完成强；这说明当前版本更偏追时间。

### 5.4 能耗分

| reward | weight | params | 对齐目标 |
|---|---:|---|---|
| `energy` | `-3.5e-5` | 无 | energy，直接惩罚 `abs(torque * joint_vel)` |
| `rough_energy` | `-2.2e-5` | rough gate | rough terrain energy |
| `joint_torques` | `-1.5e-4` | 无 | energy，降低实际扭矩 |
| `dof_vel` | `-0.00115` | 无 | energy，降低关节速度 |
| `joint_acc` | `-6e-7` | 无 | energy/pose，降低高频动作 |
| `action_rate` | `-0.022` | 无 | energy，减少动作一阶抖动 |
| `action_smoothness` | `-0.023` | 无 | energy，减少动作二阶抖动 |
| `score_guidance` | `0.35` | `tracking_std=0.62`, `posture_std=0.25`, `power_scale=55.0`, `posture_weight=0.35` | time/pose/energy 综合 |

对齐判断：

- `energy` 的原始量和平台 `energy_score` 完全一致，都是 `sum(abs(joint_vel)*abs(applied_torque))`。
- `dof_vel=-0.00115` 很强，是当前主要的节能项之一。
- `score_guidance` 的 `power_scale=55` 比平台指数尺度 `100` 更严格，会更早压低高功率动作。
- 但由于速度/推进奖励很强，能耗项可能仍被时间项压过。

### 5.5 步态、足端与接触

| reward | weight | params | 对齐目标 |
|---|---:|---|---|
| `feet_air_time` | `0.28` | `threshold=0.18` | gait/time，鼓励步态节律 |
| `feet_clearance` | `0.18` | `target_height=0.095`, `std=0.06`, `terrain_height_scale=0.85`, `max_terrain_extra_height=0.14`, `speed_height_scale=0.005` | completion/pose，通过楼梯 |
| `feet_swing_forward` | `0.08` | `target_forward=0.12`, `std=0.08`, `min_command=0.1` | completion/time，帮助上楼梯 |
| `feet_slide` | `-0.15` | 无 | energy/pose，减少滑动 |
| `feet_stumble` | `-0.1` | 无 | completion/pose，减少绊脚 |
| `air_time_variance_penalty` | `-0.55` | 无 | pose/energy，步态对称 |
| `base_lateral_vel` | `-0.42` | 无 | energy/pose，减少横漂 |
| `hip_to_default` | `-0.15` | 无 | pose/energy，保持自然髋姿态 |
| `joint_position_penalty` | `-0.11` | `stand_still_scale=2.0`, `cmd_threshold=0.1`, `ang_cmd_threshold=0.2` | pose/energy，避免极端关节 |
| `stand_still_motion` | `-0.85` | `lin_cmd_threshold=0.15`, `ang_cmd_threshold=0.2`, `vertical_vel_scale=0.5`, `ang_vel_scale=0.5`, `joint_vel_scale=0.1` | energy，低命令不乱动 |
| `commanded_still_penalty` | `-0.55` | `cmd_threshold=0.08`, `still_speed_threshold=0.12` | completion/time，有命令不能停 |
| `dof_pos_limits` | `-0.3` | 无 | stability，避免关节极限 |
| `undesired_contacts` | `-0.3` | `threshold=1` | completion，减少异常接触 |
| `termination` | `-5.0` | 无 | completion，惩罚真实失败 |

对齐判断：

- 足端奖励主要服务楼梯完成率。
- `feet_clearance` 参数较激进，能减少绊脚，但可能拉高能耗。
- `stand_still_motion` 与 `commanded_still_penalty` 成对出现，防止“省电不动”和“静止乱动”两个极端。

### 5.6 迷宫、墙体与提前转向

| reward | weight | params | 对齐目标 |
|---|---:|---|---|
| `maze_context_gate` | `0.0` | `goal_dist_gate=14.0`, `obstacle_threshold=-0.8`, `temperature=0.18`, `front_cols=10` | diagnostic |
| `maze_anticipatory_turn` | `0.85` | `obstacle_threshold=-0.72`, `front_cols=8`, `side_width=4`, `wall_start=0.2`, `wall_full=0.72`, `target_yaw_rate=0.85`, `target_forward_speed=0.95` | time/completion，提前绕墙 |
| `wall_collision` | `-12.0` | `obstacle_threshold=-0.75`, `front_cols=3`, `wall_score_threshold=0.55`, `touch_penalty=0.12`, `impact_speed=0.55`, `impact_penalty=1.6` | completion/energy，防高速撞墙 |
| `wall_stall_penalty` | `-1.2` | `front_cols=5`, `wall_score_threshold=0.38`, `still_speed=0.12` | time/completion，防贴墙卡住 |
| `wall_proximity` | `-0.35` | `front_cols=7`, `wall_score_threshold=0.18` | pose/energy，少擦墙 |
| `open_space` | `0.05` | `obstacle_threshold=-0.35`, `front_cols=8` | completion，偏向开阔区域 |
| `corridor_centering` | `-0.18` | `front_cols=8`, `wall_score_threshold=0.2`, `center_band_half_width=1` | completion/pose，走廊居中 |

对齐判断：

- `maze_anticipatory_turn` 是当前版本非常关键的时间分补丁：高速迷宫时提前转弯。
- `wall_collision=-12` 是强安全阀，避免高速策略直接撞墙。
- 墙体奖励都受 `maze_goal_dist_gate=14` 限制，避免把楼梯误判成墙。

## 6. 总体对齐评价

### 优点

- `task_complete=155`、`reach_goal=30` 明确对齐 Track 的 `completion_coeff`。
- `navigation_time=-0.022`、高分段速度、高 `goal_velocity_projection` 明确拉 `time_score`。
- `pose_score_formula`、`rough_roll_pitch_abs`、`rough_ang_vel_xy` 对齐 `pose_score`。
- `energy`、`rough_energy`、`dof_vel`、`joint_torques` 对齐 `energy_score` 原始量。
- `maze_anticipatory_turn` 与 workflow 中提前转向逻辑同步，适合高速迷宫。

### 风险

- 当前配置明显偏高速，time 驱动强于 energy 驱动。
- `maze_lin_vel_x=[1.16,1.38]` 和 `forward_heading_velocity target_speed=1.25` 可能显著拉高关节速度。
- `feet_clearance target_height=0.095`、`max_terrain_extra_height=0.14` 可能提升通过率，但也可能增加抬腿能耗。
- 如果当前分数表现是 `pose≈70, time≈72, energy≈48`，这套配置更可能优先提升 time，而不一定提升 energy。

## 7. 建议观察指标

优先看这些面板/JSON 字段：

- `completion_coeff`: 低于 0.95 时，优先完成率。
- `time_score`: 同步后理论上应上升。
- `energy_score`: 如果继续低于 50，说明高速 reward 压过能耗约束。
- `pose_score`: 若从 70 明显跌落，说明高速/迷宫提前转向引入了姿态代价。
- `reward_energy`, `reward_dof_vel`, `reward_joint_torques`: 判断能耗惩罚是否真的在变好。
- `reward_maze_anticipatory_turn`, `reward_wall_collision`, `reward_wall_stall_penalty`: 判断高速迷宫是否在用提前转向，而不是硬撞墙。

## 8. 结论

当前同步后的奖励配置是“完成率 + 时间分优先，粗糙地形和墙体逻辑兜底姿态/能耗”的设计。

它和 Track 评分标准的对齐方式是：

```text
completion_coeff: task_complete / reach_goal / approach_goal / wall & stuck penalties
time_score: high phase speed / goal_velocity_projection / forward_heading_velocity / navigation_time
pose_score: pose_score_formula / flat_orientation / ang_vel_xy / rough pose rewards
energy_score: energy / rough_energy / dof_vel / joint_torques / action smoothness
```

整体更像冲击更高 `time_score` 的版本。如果下一次评估 energy 继续低，应再把速度锚点和推进奖励往回收，而不是继续增强能耗惩罚。
