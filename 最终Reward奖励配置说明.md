# 最终 Reward 奖励配置说明

对应文件：

- `agent_ppo/conf/train_env_conf_track_nav.toml`
- `agent_ppo/feature/reward_process.py`

## 这版的核心思路

这版不再堆很多“步态外观奖励”，而是把奖励压成四条主线：

1. 强速度跟随：必须真的跟上发布的 `base_velocity`。
2. 跑速带：实际前向速度要进入更高速度区间。
3. 官方评分拟合：姿态分和能耗分给高权重。
4. 导航/迷宫：保留目标推进、墙体、开阔空间、迷宫转向等完整导航奖励。

同时去掉了 `forward_heading_velocity` 和 `backward_penalty`。原因是速度已经由 `track_lin_vel_xy` 管，方向由 `goal_velocity_projection` 管；再用 `forward_heading_velocity` 容易奖励“头朝哪就往哪冲”，和目标导航发生冲突。

## 最终启用奖励

| 奖励名 | 权重 | 作用 |
|---|---:|---|
| `track_lin_vel_xy` | `5.20` | 主速度跟随，比旧版更重、更严格。 |
| `command_speed_advantage` | `1.70` | 跟不上命令速度会明显吃亏，允许少量超速。 |
| `track_ang_vel_z` | `0.25` | 保留 yaw 跟随，帮助迷宫转向。 |
| `lin_vel_z` | `-0.45` | Go2 稳定项，压身体上下弹跳。 |
| `ang_vel_xy` | `-0.20` | Go2 稳定项，压 roll/pitch 快速晃动。 |
| `joint_acc` | `-2.5e-7` | Go2 稳定项，压关节加速度。 |
| `energy_score_formula` | `1.40` | 官方能耗分拟合，高能耗分给正奖励。 |
| `dof_pos_limits` | `-0.45` | Go2 稳定项，防止关节撞限位。 |
| `correct_base_height` | `-0.28` | Go2 稳定项，目标高度 `0.38m`，防低趴。 |
| `action_rate` | `-0.006` | Go2 稳定项，动作一阶平滑。 |
| `action_smoothness` | `-0.010` | Go2 稳定项，动作二阶平滑，重点压抖。 |
| `termination` | `-40.0` | 非超时失败惩罚。 |
| `non_completion_timeout` | `-160.0` | 未完成超时惩罚。 |
| `pose_score_formula` | `2.60` | 官方姿态分拟合，高姿态分给强奖励。 |
| `hip_to_default` | `-0.05` | Go2 稳定项，压外八/内八但不锁死腿。 |
| `feet_air_time` | `0.18` | 降权保留一点迈步信号，避免继续奖励跳。 |
| `running_speed_band` | `4.20` | 跑速带提高，目标实际速度 `3.35m/s`。 |

## 导航和迷宫奖励

| 奖励名 | 权重 | 作用 |
|---|---:|---|
| `goal_heading_alignment` | `0.90` | 机身朝向目标时给奖励。 |
| `goal_velocity_projection` | `4.20` | 只有速度投影朝目标方向才吃主要导航奖励。 |
| `goal_backtrack_penalty` | `-2.5` | 远离目标方向时扣分。 |
| `approach_goal` | `18.0` | 每步目标距离减少就奖励，是主要目标推进信号。 |
| `goal_distance` | `0.8` | 越接近目标越有平滑奖励。 |
| `reach_goal` | `1.0` | 改成进度里程碑稀疏奖励。 |
| `task_complete` | `260.0` | 进入最终 `0.6m` 完成半径的大额奖励。 |
| `navigation_time` | `-0.060` | 每步时间惩罚，催促更快完成。 |
| `maze_anticipatory_turn` | `1.20` | 迷宫前方有墙时提前向开口方向转。 |
| `wall_collision` | `-8.0` | 撞墙惩罚，速度越快越痛。 |
| `wall_stall_penalty` | `-1.2` | 墙前停住扣分。 |
| `wall_proximity` | `-0.12` | 贴墙太近扣分。 |
| `open_space` | `0.45` | 鼓励走前方开阔空间。 |
| `corridor_centering` | `-0.18` | 走廊中偏心扣分。 |
| `directed_exploration` | `0.020` | 朝目标方向探索新区域给极小奖励。 |
| `stuck_penalty` | `-1.4` | 有命令但身体几乎不动时扣分。 |

`maze_context_gate` 保留为 `0.0` 诊断项，不参与训练。

## 新的 reach_goal 里程碑

`reach_goal` 不再只是 `0.6m` 终点半径奖励，而是新增“从起点向终点累计推进距离”的一次性稀疏奖励：

```toml
[rewards.reach_goal]
weight = 1.0
[rewards.reach_goal.params]
threshold = 0.6
milestones = [14.0, 18.0, 25.0, 30.0, 40.0]
milestone_rewards = [10.0, 14.0, 20.0, 28.0, 40.0]
final_reward = 80.0
```

含义：

- 推进 14m：给 `10`
- 推进 18m：给 `14`
- 推进 25m：给 `20`
- 推进 30m：给 `28`
- 推进 40m：给 `40`
- 进入终点 `0.6m`：额外给 `80`

每个里程碑每个 episode 只给一次。这样比只等终点稀疏奖励更容易学到“持续向终点推进”。

## 已去掉的主要奖励

这些奖励不再出现在当前 TOML reward 区：

- `forward_heading_velocity`
- `backward_penalty`
- `energy`
- `joint_torques`
- `flat_orientation`
- `posture_stability`
- `joint_position_penalty`
- `stand_still_motion`
- `commanded_still_penalty`
- `feet_slide`
- `feet_stumble`
- `dof_vel`
- `base_lateral_vel`
- `air_time_variance_penalty`
- `feet_clearance`
- `feet_swing_forward`
- `running_stride_span`
- `running_contact_pattern`
- `running_air_time_band`
- `running_speed_efficiency`
- `score_guidance`
- `track_score_balance`
- 非迷宫靠边/平路路线奖励
- 粗糙地形额外奖励

这次去掉它们的目的不是说它们永远没用，而是先避免奖励互相打架，让模型先围绕“速度跟随 + 目标推进 + 官方姿态/能耗分”学习。

## 为什么这样改

旧版的问题是：`feet_air_time`、`running_air_time_band`、`feet_clearance`、`running_contact_pattern` 等项会奖励“跑步的外观特征”，但模型可以用跳、低趴、乱抬腿去满足这些特征。

新版把这些外观奖励大幅减少，只保留很小的 `feet_air_time`。真正的大头变成：

```text
必须跟速度：track_lin_vel_xy + command_speed_advantage
必须快：running_speed_band
必须朝目标有效推进：goal_velocity_projection + approach_goal
必须评分好：pose_score_formula + energy_score_formula
必须别乱跳：lin_vel_z + correct_base_height + action_smoothness
```

这样目标更直接：不是“看起来像跑”，而是“高速、朝目标、姿态和能耗分都高”。
