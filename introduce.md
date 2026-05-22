# 四足机器人运动控制 开发指南

> **文档版本**: 22.0.3  
> **来源**: 腾讯开悟平台 (tencentarena.com)  
> **竞赛**: 四足机器人自主导航运控赛题

---

## 目录

- [项目简介](#项目简介)
- [环境详述](#环境详述)
- [智能体详述](#智能体详述)
- [数据协议](#数据协议)
- [腾讯开悟强化学习框架](#腾讯开悟强化学习框架)
- [监控与日志](#监控与日志)
- [强化学习系统系列技术标准](#强化学习系统系列技术标准)

---

## 项目简介

### 任务目标

四足机器人自主导航运控赛题。选手需要使用强化学习算法训练智能体，让其控制GO2机器狗在仿真环境内探索并学习自主导航与运动控制策略，使得机器狗可以在 **未知/半未知** 场景中实现 **自主寻路**，以尽可能短的时间跨越地形，同时保持运动的稳定性。

任务包含两种模式：

- **标准模式（Standard）**：机器人分别在多种复杂地形（坡面、楼梯、迷宫）上行走，以前进距离、通过时间、能量效率和姿态稳定性综合评分
- **赛道模式（Track）**：机器人在由多种子地形串联构成的赛道上从起点导航至终点，以完成数量、通过时间、姿态稳定性和能量效率综合评分

选手需要设计和优化强化学习算法，使四足机器人能够：

- 在各类地形上保持稳定行走（运动控制）
- 自主规划路径并避开障碍物（自主导航）
- 高效、快速地从起点到达终点（赛道模式）

### 环境介绍

#### 地形

赛题使用 trimesh 类型地形，分为内置地形和自研地形两类。standard 模式下各子地形按比例分配；track 模式下子地形串联成单向赛道。

##### 标准模式地形

| 地形类型 | 说明 | 图例 |
|---|---|---|
| pyramid_slope | 金字塔坡面，向上 | (图示) |
| pyramid_slope_inv | 金字塔坡面，向下 | (图示) |
| pyramid_stairs | 金字塔楼梯，向上 | (图示) |
| pyramid_stairs_inv | 金字塔楼梯，向下 | (图示) |
| maze | 迷宫地形，随机生成高度为 0.5m 的障碍物，地形中至少保持一条从进入边到对边的通路 | (图示) |

##### 赛道模式地形

| 地形类型 | 说明 | 图例 |
|---|---|---|
| track | 由多种子地形串联构成的单向赛道：前段为 pyramid_slope、pyramid_slope_inv、pyramid_stairs、pyramid_stairs_inv 的任意排列，末段必须为 open_entry_maze（赛道终点恒为迷宫） | (图示) |
| open_entry_maze | 赛道专用终点地形：迷宫出入口均为开放通道，机器人从进入边行至对边的出口即完成赛道 | (图示) |

#### 元素介绍

| 元素 | 说明 |
|---|---|
| Go2 四足机器人 | Unitree Go2 四足机器人，具有 4 条腿 × 3 关节（hip/thigh/calf）共 12 个可控关节 |
| 地形 | 多种复杂地形（坡面、楼梯、迷宫、赛道），用于测试机器人的运动能力和导航能力 |

> 在创建训练任务和评估任务时，上述地形的配置方式有所不同。具体请查看开发指南-环境配置部分。

### 计分规则

#### Standard 模式

以最稳定、最节能的方式尽可能走远（以走穿地形为目标）。

$$
\text{总分} = 0.4 \times \text{Score}_{\text{forward}} + 0.2 \times \text{Score}_{\text{time}} + 0.2 \times \text{Score}_{\text{energy}} + 0.2 \times \text{Score}_{\text{posture}}
$$

| 子项 | 权重 | 含义 |
|---|---|---|
| 前进距离分数 | 0.4 | 从地形块中心出生，按"与出生点的 2D 欧氏距离 / 半块长度"归一化；走到半块即满分（= 走穿：从中心开始穿过半个地形长度） |
| 时间分数 | 0.2 | 仅走穿地形的模型获得，用时越短分越高 |
| 能耗分数 | 0.2 | episode 平均单步关节机械功率的指数衰减，越节能越高 |
| 姿态分数 | 0.2 | episode 平均 roll/pitch 偏移的指数衰减，越平稳越高 |

#### Track 模式

在限定步数内，以最快、最稳定、最节能的方式从起点到达目标点。

$$
\text{总分} = \text{完成系数} \times \left( 0.4 \times \text{Score}_{\text{time}} + 0.4 \times \text{Score}_{\text{posture}} + 0.2 \times \text{Score}_{\text{energy}} \right)
$$

其中**完成系数**为当前任务中完成赛道的机器狗数量占总数的比例：单只机器狗完成赛道记为 1、失败或超时记为 0，对批次内所有机器狗求均值即可。例如并行 1024 只机器狗、其中 512 只走到终点，完成系数 = 0.5。

| 子项 | 权重 | 含义 |
|---|---|---|
| 时间分数 | 0.4 | 用时越短分越高 |
| 姿态分数 | 0.4 | episode 平均 roll/pitch 偏移的指数衰减 |
| 能耗分数 | 0.2 | episode 平均单步关节机械功率的指数衰减 |

#### 注意事项

- 任务失败条件：主体或关节接触地面（姿态异常 / 摔倒）
- 任务超时条件：达到最大步数（`episode_length_s` 对应的步数）仍未完成
- Standard 模式走穿判定：从地形块中心出生，$\| \text{pos}_{\text{current}} - \text{pos}_{\text{spawn}} \|_2 \geq L_{\text{terrain}} / 2 - 0.1$（2D 欧氏距离，方向无关）；默认 $L_{\text{terrain}} = 8\,\text{m}$，阈值约 3.9 m
- Track 模式完成判定：机器人从起点到达终点（目标点）
- Sim 赛段：使用 height scan 观测，无需深度视觉相机

---

## 环境详述

### 环境配置

在智能体和环境的交互中，首先会调用 `env.reset` 方法，该方法接受一个 `usr_conf` 参数，通过读取 `train_env_conf_standard_locomotion.toml` 文件的内容来实现定制化的环境配置。用户可以通过修改该 TOML 文件中的内容来调整环境配置。

```python
# usr_conf 为用户传入的环境配置
reset_data = env.reset(usr_conf)
obs, critic_obs = reset_data
```

`train_env_conf_standard_locomotion.toml` 为示例环境配置，用于在标准模式训练四足机器人的运动控制能力。

环境配置中包含以下配置信息：

| 配置项 | 类型 | 合法范围 | 说明 |
|---|---|---|---|
| `env.num_envs` | int | [1, 4096] | 并行环境数量 |
| `env.episode_length_s` | float | >0 | 最大 episode 时长（秒） |
| `terrain.mode` | string | "standard" \| "track" | 地形模式 |
| `terrain.num_rows` | int | [1, 10] | 难度级别数（沿 X 轴的课程档位） |
| `terrain.num_cols` | int | [1, 40] | 同一难度下的并行地块数（沿 Y 轴） |
| `terrain.difficulty_range` | list[float] | [0.0, 1.0] | 难度范围 |
| `terrain.curriculum` | bool | true/false | 是否启用地形课程学习 |
| `terrain.max_init_terrain_level` | int | [0, 9] | 机器人初始放置的最大难度档 |
| `terrain.standard.*.proportion` | float | [0, 1] | 各子地形比例（总和须为 1.0） |
| `domain_rand.enable_domain_rand` | bool | true/false | 域随机化总开关 |
| `domain_rand.randomize_friction` | bool | true/false | 是否随机化地面摩擦系数 |
| `domain_rand.friction_range` | list[float] | ≥0 | 摩擦系数采样范围 |
| `domain_rand.push_robots` | bool | true/false | 是否周期性施加外部推力 |
| `domain_rand.push_interval_s` | float | >0 | 推力间隔（秒） |
| `domain_rand.max_push_vel_xy` | float | ≥0 | XY 平面最大推力速度 (m/s) |
| `noise.add_noise` | bool | true/false | 是否在观测中加入噪声 |
| `init_state.pos` | list[float] | z: [0.30, 0.60] | 机器人初始位置 [x, y, z] (m) |
| `commands.resampling_time` | list[float] | >0 | 速度命令重采样区间 [min, max]（秒） |
| `commands.limit.lin_vel_x` | list[float] | — | X 方向线速度采样上限 |
| `commands.limit.lin_vel_y` | list[float] | — | Y 方向线速度采样上限 |
| `commands.limit.ang_vel_z` | list[float] | — | 偏航角速度采样上限 |
| `commands.ranges.lin_vel_x` | list[float] | — | X 方向线速度初始采样范围 |
| `commands.ranges.lin_vel_y` | list[float] | — | Y 方向线速度初始采样范围 |
| `commands.ranges.ang_vel_yaw` | list[float] | — | 偏航角速度初始采样范围 |
| `rewards.*.weight` | float | — | 各奖励项权重 |
| `rewards.*.params.*` | — | — | 各奖励项参数 |

> **补充说明**：
> - `train_env_conf_standard_locomotion.toml` 文件中的配置仅在训练时生效，请按上表数据描述进行配置。若配置错误，训练任务会变为"失败"状态，此时可以通过查看 env 模块的错误日志进行排查。
> - 若需调整模型评估任务时的配置，用户需要通过腾讯开悟平台创建评估任务并完成环境配置，详细参数见[智能体模型评估模式](#模型评估模式)。

#### train_env_conf_standard_locomotion.toml 默认配置

```toml
[env]
num_envs = 2048
episode_length_s = 25

[env_conf]
seed = 0

[terrain]
mode = "standard"
num_rows = 10
num_cols = 20
difficulty_range = [0.0, 1.0]
curriculum = true
max_init_terrain_level = 5

[terrain.standard.pyramid_slope]
proportion = 0.15

[terrain.standard.pyramid_slope_inv]
proportion = 0.2

[terrain.standard.pyramid_stairs]
proportion = 0.25

[terrain.standard.pyramid_stairs_inv]
proportion = 0.3

[terrain.standard.maze]
proportion = 0.1

[domain_rand]
enable_domain_rand = true
randomize_friction = true
friction_range = [0.3, 1.5]
push_robots = true
push_interval_s = 15
max_push_vel_xy = 0.5

[noise]
add_noise = true

[init_state]
pos = [0.0, 0.0, 0.35]

[commands]
resampling_time = [10.0, 10.0]

[commands.limit]
lin_vel_x = [-2.0, 2.0]
lin_vel_y = [-1.5, 1.5]
ang_vel_z = [-1.5, 1.5]

[commands.ranges]
lin_vel_x = [0.0, 0.5]
lin_vel_y = [-0.3, 0.3]
ang_vel_yaw = [-1.0, 1.0]

[rewards.track_lin_vel_xy]
weight = 1.0
[rewards.track_lin_vel_xy.params]
std = 0.25
command_name = "base_velocity"

[rewards.track_ang_vel_z]
weight = 0.5
[rewards.track_ang_vel_z.params]
std = 0.25
command_name = "base_velocity"

[rewards.lin_vel_z]
weight = -2.0

[rewards.ang_vel_xy]
weight = -0.05

[rewards.joint_acc]
weight = -2.5e-7

[rewards.joint_torques]
weight = -1e-4

[rewards.dof_pos_limits]
weight = -2.0

[rewards.action_rate]
weight = -0.01

[rewards.undesired_contacts]
weight = -1.0
[rewards.undesired_contacts.params]
threshold = 1

[rewards.flat_orientation]
weight = -1.5

[rewards.reach_goal]
weight = 10.0
[rewards.reach_goal.params]
threshold = 0.6
```

> **Standard模式下的合法子地形类型**：`pyramid_slope` | `pyramid_slope_inv` | `pyramid_stairs` | `pyramid_stairs_inv` | `maze`

#### 切换到 Track 模式

进入导航阶段训练时，需把 terrain 段替换为 track 配置，并参考 LocomotionConfig 自行设计新的训练阶段配置。

```toml
[terrain.track]
track_length = 5
sub_terrains = ["pyramid_slope", "pyramid_slope_inv", "pyramid_stairs", "pyramid_stairs_inv", "open_entry_maze"]
```

> **Track模式下的合法子地形类型**：`pyramid_slope` | `pyramid_slope_inv` | `pyramid_stairs` | `pyramid_stairs_inv` | `open_entry_maze`
> 
> 需要注意 `open_entry_maze` 必须配在赛道最后，否则训练会报错。

### 环境信息

| 数据名 | 数据类型 | 数据描述 |
|---|---|---|
| frame_no | int | 当前交互帧号 |
| obs | torch.Tensor | 策略观测 (num_envs, obs_dim) |
| rewards | torch.Tensor | 当前步总 reward (num_envs,) |
| terminated | torch.Tensor[bool] | 真实终止（摔倒、目标达成） |
| truncated | torch.Tensor[bool] | 超时截断 |
| infos | dict | Isaac Lab / RSL-RL extras |
| privileged_obs | torch.Tensor \| None | critic 观测 (num_envs, critic_obs_dim) |

#### 奖励信息（reward）

reward 是 Isaac Lab 按 TOML 配置的 `[rewards.*]` 段实时计算出的每一步奖励总和，shape = (num_envs,)。

> 注意：reward 是强化学习训练的驱动信号，由代码包默认激活的 11 项 reward（见"环境监控信息-Reward 指标"一节）加权求和得到。它不等于平台评分系统的"总分"——总分由默认的监控信息进行上报。

#### 观测信息（observation）

策略观测 obs 是传递给 Actor 网络的输入，布局如下：

```
obs = [proprio(45) | height_scan(256)] → 301 维
```

**proprio（45 维）字段布局**：

| 区间 | 维度 | 含义 | 来源 |
|---|---|---|---|
| [0:3] | 3 | base_ang_vel，机体角速度，scale=0.25 | Isaac Lab mdp |
| [3:6] | 3 | projected_gravity，重力方向投影 | Isaac Lab mdp |
| [6:9] | 3 | velocity_commands (vx, vy, wz) | command manager |
| [9:21] | 12 | joint_pos_rel，关节相对默认位置 | robot data |
| [21:33] | 12 | joint_vel_rel，关节速度，scale=0.05 | robot data |
| [33:45] | 12 | last_action，上一帧动作 | action manager |

**height_scan（256 维）**：

| 字段名 | 区间 | 类型 | 说明 |
|---|---|---|---|
| height_scan | obs[:, 45:301] | torch.Tensor | 16×16 前方高度扫描，clip [-5, 5]，scale=2.5 |

**privileged_obs（316 维）**是传递给 Critic 网络的观测，在 proprio 基础上额外包含 base_lin_vel（机体线速度）和 joint_effort（关节力矩）等特权信息，仅训练时使用，体现"不对称 Actor-Critic"设计。

> **补充说明**：track 地形下环境额外提供 `env.goal_positions`（目标点世界坐标）和 `env.goal_yaw`（目标点朝向），以及 `env.scene.sensors["nav_scanner"]`（前瞻遮挡扫描），选手可从这些属性构造导航特征并拼接到 obs。目标点观测详细信息请参考：Go2 SDK 开发指南

#### 额外信息（infos）

infos 是一个 dict，包含仿真环境给的辅助信息。

#### 动作空间

Go2 为 12 自由度四足机器人，动作空间为 12 维连续动作：

| 字段名 | 类型 | Shape | 取值范围 | 说明 |
|---|---|---|---|---|
| actions | float32 | (num_envs, 12) | [-1.0, 1.0] | 12 个关节控制动作 |

**合法动作**：

动作值为归一化的偏移量，经过 action_scale（默认 0.25）缩放后加到默认关节角度上，作为 PD 控制器的目标角度。关节维度对应：

| 维度 | 关节组 | 说明 |
|---|---|---|
| 0~2 | 前左腿 | hip / thigh / calf |
| 3~5 | 前右腿 | hip / thigh / calf |
| 6~8 | 后左腿 | hip / thigh / calf |
| 9~11 | 后右腿 | hip / thigh / calf |

> 注意：具体关节顺序以 Isaac Lab Unitree Go2 资产配置为准。

#### 时间信息

步（step）和帧（frame）存在一一对应关系。

每一步中，智能体选择一个 12 维动作，环境据此更新状态并返回新的观测、奖励和终止信号。`env.step()` 返回的 `frame_no` 即当前交互帧号。

### 环境监控信息

监控面板中包含 env 模块，表示环境指标数据，每 1 分钟采集最新结束的 episode 数据、求平均后展示。详细说明如下：

#### Standard 模式

##### 全局环境指标

| 面板名称 | 指标名称 | 说明 |
|---|---|---|
| 已结束任务数 | completed_count | 正常完成的 episode 数 |
| 已结束任务数 | abnormal_count | 异常终止的 episode 数 |
| 已结束任务数 | timeout_count | 超时终止的 episode 数 |
| 得分 | total_score | 单局总分均值 |
| 得分 | distance_score | 单局前进距离分均值 |
| 得分 | time_score | 单局时间分均值 |
| 得分 | energy_score | 单局能耗分均值 |
| 得分 | pose_score | 单局姿态分均值 |
| 步数 | step | 单局平均步数 |

##### [terrain_type] 指标（按地形类型分 Tab）

| 面板名称 | 指标命名规律 | 说明 |
|---|---|---|
| 地形-完成数 | completed_count_[terrain_type] | 该地形正常完成 episode 数 |
| 地形-失败数 | abnormal_count_[terrain_type] | 该地形异常终止 episode 数 |
| 地形-超时数 | timeout_count_[terrain_type] | 该地形超时终止 episode 数 |
| 地形-总分 | total_score_[terrain_type] | 该地形总分均值 |
| 地形-距离分数 | distance_score_[terrain_type] | 该地形前进距离分均值 |
| 地形-时间分数 | time_score_[terrain_type] | 该地形时间分均值 |
| 地形-能耗分数 | energy_score_[terrain_type] | 该地形能耗分均值 |
| 地形-姿态分数 | pose_score_[terrain_type] | 该地形姿态分均值 |
| 地形-步数 | step_[terrain_type] | 该地形平均步数 |

> `[terrain_type]` 需替换为具体地形名称（如 pyramid_slope、maze 等），多种地形对应多个 Tab。

#### Track模式

##### 全局环境指标

| 面板名称 | 指标名称 | 说明 |
|---|---|---|
| 已结束任务数 | completed_count | 正常完成的 episode 数 |
| 已结束任务数 | abnormal_count | 异常终止的 episode 数 |
| 已结束任务数 | timeout_count | 超时终止的 episode 数 |
| 得分 | total_score | 单局总分均值 |
| 得分 | energy_score | 单局能耗分均值 |
| 得分 | pose_score | 单局姿态分均值 |
| 得分 | time_score | 单局时间分均值（底层指标 key 为 kaiwu_step_score） |
| 步数 | step_avg | 单局平均步数 |
| Reward 均值 | reward_mean | 单局平均奖励 |
| Reward 均值 | reward_std | 单局奖励标准差 |

##### Track 赛道-难度档

| 面板名称 | 指标命名规律 | 说明 |
|---|---|---|
| 赛道-完成数 | completed_count_track_l{0~9} | 各难度档正常完成 episode 数 |
| 赛道-失败数 | abnormal_count_track_l{0~9} | 各难度档异常终止 episode 数 |
| 赛道-超时数 | timeout_count_track_l{0~9} | 各难度档超时终止 episode 数 |
| 赛道-总分 | total_score_track_l{0~9} | 各难度档总分均值 |
| 赛道-能耗分数 | energy_score_track_l{0~9} | 各难度档能耗分均值 |
| 赛道-姿态分数 | pose_score_track_l{0~9} | 各难度档姿态分均值 |
| 赛道-时间分数 | time_score_track_l{0~9} | 各难度档时间分均值 |

#### Reward 指标

代码包默认激活的 reward 对应的监控面板：

| 面板名称 | 指标名称 | 说明 |
|---|---|---|
| 线速度跟踪奖励 | reward_track_lin_vel_xy | XY 速度命令跟踪奖励均值 |
| 角速度跟踪奖励 | reward_track_ang_vel_z | yaw 角速度命令跟踪奖励均值 |
| 安全奖励 | reward_undesired_contacts | 非脚掌接触惩罚均值 |
| 安全奖励 | reward_dof_pos_limits | 关节位置极限惩罚均值 |
| 平坦姿态奖励 | reward_flat_orientation | 非直立姿态惩罚均值 |
| 到达目标 | reward_reach_goal | 到达目标点奖励均值（仅 track 地形且 env.goal_positions 被维护时生效） |

> 选手若在 `reward_process.py` 新增了 reward，需要在 `agent_ppo/conf/monitor_builder.py` 中参考 Group 2 的示例自行添加对应面板。

---

## 智能体详述

我们在代码包中提供了智能体的简单实现，本文将对该部分内容进行讲解，包括特征处理和奖励处理等。

### 观测处理

环境返回的 observation 信息包含了针对智能体的局部观测，可以在 `PolicyObservationProcess` 中进行处理。代码包默认直接透传 `self.default_observation()` 返回的 301 维张量，选手可在此基础上做特征工程（归一化、聚合、拼接额外特征等）：

```python
class PolicyObservationProcess(ObservationProcess):
    target_group = "policy"

    def process(self):
        obs = self.default_observation()
        # TODO (track 地形)：可按需从 env.goal_positions / env.goal_yaw
        # 或 env.scene.sensors["nav_scanner"] 构造特征并拼接到 obs。
        return obs
```

观测数据布局：

| 区间 | 维度 | 含义 | 来源 |
|---|---|---|---|
| [0:3] | 3 | base_ang_vel，机体角速度，scale=0.25 | Isaac Lab mdp |
| [3:6] | 3 | projected_gravity，重力方向投影 | Isaac Lab mdp |
| [6:9] | 3 | velocity_commands，速度命令 (vx, vy, wz) | command manager |
| [9:21] | 12 | joint_pos_rel，关节相对默认位置 | robot data |
| [21:33] | 12 | joint_vel_rel，关节速度，scale=0.05 | robot data |
| [33:45] | 12 | last_action，上一帧动作 | action manager |
| [45:301] | 256 | height_scan，16×16 前方高度扫描 | height_scanner |

**Track 地形扩展**：环境额外提供 `env.goal_positions`（出口位置）、`env.goal_yaw`（出口朝向）和 `env.scene.sensors["nav_scanner"]`（前瞻遮挡扫描）。

### 特征处理

当前代码包中 `PolicyObservationProcess` 直接使用 `default_observation()` 返回的原始观测，未做额外特征工程处理。

```python
def process(self):
    obs = self.default_observation()
    return obs
```

选手可以在此基础上进行特征工程扩展。

### 奖励处理

代码包提供了两个示例奖励函数，位于 `reward_process.py`：

| 奖励名称 | 作用 | 类型 |
|---|---|---|
| `_reward_reach_goal` | 机器人与出口距离小于 0.6 m 时给予 1.0 的稀疏奖励 | sparse |
| `_reward_forward_velocity` | 鼓励机器人沿本体 x 轴方向前进的 dense 奖励 | dense |

其余通用 locomotion 奖励继承自 `RewardProcessBase`，在 TOML 配置中激活 `[rewards.<name>]` 即可，无需重复实现。选手若需训练导航策略，请在 `reward_process.py` 自行添加更多奖励函数。

### 算法介绍

代码包提供了基于 PPO（Proximal Policy Optimization）的 Actor-Critic 算法实现。

| 算法 | 文件路径 | 说明 |
|---|---|---|
| PPO + ActorCritic | model/actor_critic.py | 扁平 MLP Actor-Critic 网络，Gaussian 动作分布 |

#### ActorCritic 网络结构

采用独立的 Actor 和 Critic 两个 MLP 网络：

| 组件 | 输入维度 | 隐藏层 | 输出维度 | 特殊设计 |
|---|---|---|---|---|
| Actor | 301（policy obs） | [512, 256, 128] | 12（关节动作） | ELU 激活 |
| Critic | 316（critic obs） | [512, 256, 128] | 1（状态价值） | LayerNorm + ELU 激活 |

#### 训练超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| lr | 3e-4 | 学习率 |
| num_learning_epochs | 5 | 每次更新的 epoch 数 |
| num_mini_batches | 4 | mini-batch 数量 |
| num_steps_per_env | 48 | 每个环境采集步数 |
| model_save_interval | 500 | 模型保存间隔（迭代次数） |

### 算法监控信息

算法上报了 reward 等指标，用户可以通过腾讯开悟平台/客户端的监控功能查看。

针对当前算法的指标说明如下：

#### 算法指标组

| 面板名称 | 指标 Key | 类型 | 说明 |
|---|---|---|---|
| 总损失 | total_loss | line | PPO 总 loss（policy_loss + value_loss - entropy_loss） |
| 价值损失 | value_loss | line | Critic 网络价值函数损失 |
| 策略损失 | policy_loss | line | Actor 网络策略梯度损失 |
| 熵损失 | entropy_loss | line | 策略熵，衡量探索程度 |

#### 奖励指标组

| 面板名称 | 指标 Key | 类型 | 说明 |
|---|---|---|---|
| 线速度跟踪奖励 | reward_track_lin_vel_xy | line | XY 平面速度命令跟踪奖励 |

> 说明：以上为代码包 `monitor_builder.py` 中定义的自定义指标。其余通用 reward 指标（如速度跟踪、姿态、步态等）由 `tools/conf/monitor_default.yaml` 和 `tools/conf/monitor_default_track.yaml` 负责展示。

### 模型保存限制策略

为了避免用户保存模型的频率过于频繁，开悟平台对模型保存会有安全限制，不同的任务会有不同的限制，限制规则详情如下：

| 参数 | 值 | 说明 |
|---|---|---|
| model_save_interval | 500 | 每 500 次训练迭代保存一次模型 |

> 注意：选手可在 `conf.py` 中修改 `model_save_interval` 的值调整保存频率，但平台侧可能有额外的最小间隔限制。

### 模型评估模式

用户可以在腾讯开悟平台上创建模型评估任务。创建任务时，可以对该任务的环境进行配置，配置信息如下：

```toml
[env]
# Number of parallel environments | 并行环境数量
num_envs = 16
# Episode length in seconds | 单回合时长(秒)
episode_length_s = 20.0
# Task type: standard / track | 任务类型
task = "standard"

[terrain]
# Terrain mode: standard / track | 地形模式
mode = "standard"
# Difficulty levels to evaluate (0=easiest, 9=hardest)
# standard 模式下: num_rows = max(level)+1 自动推导, num_cols = len(sub_terrains)
# track    模式下: num_parallel_tracks 默认 10; 仅当 max(level)+1 > 10 时才自动扩展
# 评估的难度档列表(0最简单, 9最难)
level = [0, 1, 3, 4]
# Force-disable curriculum for deterministic evaluation | 强制关闭课程学习
curriculum = false

# Standard mode: deterministic sub-terrain list (no proportion)
# Each terrain occupies one column; env_i → (level[(i//n_sub)%n_lv], sub[i%n_sub])
# Standard 模式子地形列表, 每种地形占一列, env 按笛卡尔积依次放置
[terrain.standard]
sub_terrains = ["pyramid_slope", "pyramid_slope_inv", "pyramid_stairs", "pyramid_stairs_inv"]

# Track mode: linear course config (used only when mode = "track")
# Track 模式赛道配置(仅当 mode = "track" 时生效)
[terrain.track]
# Number of sub-terrains chained along the track | 赛道串联的子地形数量
track_length = 5
# Sub-terrain sequence along the track (length must equal track_length, no duplicates)
# Allowed: pyramid_slope | pyramid_slope_inv | pyramid_stairs | pyramid_stairs_inv | open_entry_maze
# 赛道上子地形顺序(长度=track_length, 不可重复)
sub_terrains = ["pyramid_slope", "pyramid_slope_inv", "pyramid_stairs", "pyramid_stairs_inv", "open_entry_maze"]
# Parallel tracks along Y-axis | Y 轴并行赛道数
# Default: 10. Each robot is placed on the track whose column index matches
# its assigned level from `[terrain] level = [...]`. The framework only
# auto-extends when max(level)+1 > 10.
# 默认 10 条并行赛道; 每只机器人按 [terrain] level 列表分配到对应 col 的赛道上;
# 仅当 max(level)+1 > 10 时框架才自动扩展 num_parallel_tracks。
num_parallel_tracks = 10

# Velocity command limits | 速度指令上限
[commands]

[commands.limit]
# Linear velocity X range (m/s) | X 方向线速度范围
lin_vel_x = [0.0, 1.0]
# Linear velocity Y range (m/s) | Y 方向线速度范围
lin_vel_y = [-0.0, 0.0]
# Angular velocity Z range (rad/s) | Z 轴角速度范围
ang_vel_z = [0.0, 0.0]
```

> ⚠️ **注意**：
> - 评估时部分训练配置（如域随机化、观测噪声）默认关闭，以确保评估结果的一致性和可复现性。
> - `task` 必须与 `mode` 保持一致；若不一致，本轮评估结果没有意义（评分字段与地形语义错位）。
> - `num_envs` 配置小于等于 16 时会产出 mp4 视频用于可视化观察模型表现，大于 16 时将不会产出 mp4 视频。



---

## 数据协议

为了方便同学们调用原始数据和特征数据，下面提供了协议供大家查阅。

> **注意**：基于 Isaac Lab 仿真环境，所有观测和动作数据均以 torch.Tensor 格式在 GPU 上传递，而非传统的字典协议格式。

### 环境交互接口

#### 环境重置（Reset）

```python
# 环境重置
reset_data = env.reset(usr_conf)

# 成功时
obs, critic_obs = reset_data
# obs: torch.Tensor, shape=(num_envs, obs_dim)       — policy 观测
# critic_obs: torch.Tensor, shape=(num_envs, critic_obs_dim) — critic 观测

# 失败时
reset_data = None
```

#### 环境步进（Step）

```python
data = env.step(actions)
frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs) = data
dones = terminated | truncated
```

**reset() 返回结构**：

| 字段名 | 字段路径 | 类型 | 取值范围 | 说明 |
|---|---|---|---|---|
| obs | reset()[0] | torch.Tensor | (num_envs, obs_dim) | policy 观测 |
| critic_obs | reset()[1] | torch.Tensor | (num_envs, critic_obs_dim) | critic 观测 |

**step() 返回结构**：

| 字段名 | 字段路径 | 类型 | 取值范围 | 说明 |
|---|---|---|---|---|
| frame_no | step()[0] | int | >=1 | 当前交互帧号 |
| obs | step()[1] | torch.Tensor | (num_envs, obs_dim) | 下一步 policy 观测 |
| rewards | step()[2] | torch.Tensor | (num_envs,) | 当前步总 reward |
| terminated | step()[3] | torch.Tensor[bool] | (num_envs,) | 真实终止 |
| truncated | step()[4] | torch.Tensor[bool] | (num_envs,) | 超时截断 |
| infos | step()[5][0] | dict | — | Isaac Lab extras |
| privileged_obs | step()[5][1] | torch.Tensor | (num_envs, critic_obs_dim) | critic 观测 |

### 观测数据协议（Observation）

#### 策略观测（Policy Observation）

```
obs = [proprio(45) | height_scan(256)]  →  301 维
```

**proprio（45 维）字段布局**

| 区间 | 维度 | 含义 | 来源 |
|---|---|---|---|
| [0:3] | 3 | base_ang_vel，机体角速度，scale=0.25 | Isaac Lab mdp |
| [3:6] | 3 | projected_gravity，重力方向投影 | Isaac Lab mdp |
| [6:9] | 3 | velocity_commands (vx, vy, wz) | command manager |
| [9:21] | 12 | joint_pos_rel，关节相对默认位置 | robot data |
| [21:33] | 12 | joint_vel_rel，关节速度，scale=0.05 | robot data |
| [33:45] | 12 | last_action，上一帧动作 | action manager |

**height_scan（256 维）**

| 字段名 | 字段路径 | 类型 | 取值范围 | 说明 |
|---|---|---|---|---|
| height_scan | obs[:, 45:301] | torch.Tensor | clip [-5, 5]，scale=2.5 | 16x16 前方高度扫描，256 条射线 |

#### Track 地形附加原始张量（选手自行消费）

代码包的 `default_observation()` 返回 301 维 obs，不内置目标点或通行性特征。当 `terrain.mode = "track"` 时，环境额外暴露以下原始张量，选手需要在 `policy_observation_process.py::process()` 里自行读取并拼接到 obs，同时同步修改 `critic_observation_process.py`、模型输入维度和 TOML 里的 obs 维度：

| 字段 | 类型 | Shape | 含义 |
|---|---|---|---|
| env.goal_positions | torch.Tensor | (num_envs, 3) | 目标点（迷宫出口）在世界坐标系下的 (x, y, z) |
| env.goal_yaw | torch.Tensor | (num_envs,) | 目标点在世界坐标系下的朝向（rad） |
| env.scene.sensors["nav_scanner"] | RayCaster | — | 前瞻遮挡扫描传感器，范围比 height_scanner 更大，适合避障 / 转向判断 |

这些都是原始值；如需构造相关新的特征处理，由选手自行设计与实现。

#### 评论家观测（Critic Observation）

```
critic_obs = [critic_proprio(60) | height_scan(256)]  →  316 维
```

**critic_proprio（60 维）字段布局**

| 区间 | 维度 | 含义 |
|---|---|---|
| [0:3] | 3 | base_lin_vel，机体线速度 |
| [3:6] | 3 | base_ang_vel，机体角速度 |
| [6:9] | 3 | projected_gravity |
| [9:12] | 3 | velocity_commands |
| [12:24] | 12 | joint_pos_rel |
| [24:36] | 12 | joint_vel_rel |
| [36:48] | 12 | joint_effort，关节力矩 |
| [48:60] | 12 | last_action |

> **补充说明**：critic_proprio（60 维）相比 proprio（45 维）额外包含 base_lin_vel 和 joint_effort 等特权数据，仅供 Critic 网络训练使用，体现"不对称 Actor-Critic"设计。

### 动作数据协议（Action）

```python
actions = torch.Tensor  # shape: (num_envs, 12)
```

| 字段名 | 类型 | Shape | 取值范围 | 说明 |
|---|---|---|---|---|
| actions | float32 | (num_envs, 12) | [-1.0, 1.0] | 12 个关节控制动作 |

#### 关节动作维度

Go2 为 12 自由度四足机器人，4 条腿 x 3 个关节：

| 维度 | 关节组 | 说明 |
|---|---|---|
| 0~2 | 前左腿 | hip / thigh / calf |
| 3~5 | 前右腿 | hip / thigh / calf |
| 6~8 | 后左腿 | hip / thigh / calf |
| 9~11 | 后右腿 | hip / thigh / calf |

> 注意：动作值为归一化偏移量，经 action_scale（默认 0.25）缩放后加到默认关节角度上，作为 PD 控制器目标角度。

### 奖励数据协议（Reward）

| 字段名 | 类型 | Shape | 说明 |
|---|---|---|---|
| rewards | float32 | (num_envs,) | 当前步总 reward |

#### 代码包默认激活的奖励项

以下 reward 已在 `train_env_conf_standard_locomotion.toml` 的 `[rewards.*]` 段激活：

| 奖励名称 | 作用 | 类型 |
|---|---|---|
| track_lin_vel_xy | 奖励 XY 速度跟踪 | 正奖励 |
| track_ang_vel_z | 奖励 yaw 角速度跟踪 | 正奖励 |
| lin_vel_z | 惩罚 Z 方向跳动 | 负奖励 |
| ang_vel_xy | 惩罚 roll/pitch 角速度 | 负奖励 |
| joint_acc | 惩罚关节加速度 | 负奖励 |
| joint_torques | 惩罚关节扭矩 | 负奖励 |
| dof_pos_limits | 惩罚接近关节极限 | 负奖励 |
| action_rate | 惩罚动作帧间变化率 | 负奖励 |
| undesired_contacts | 惩罚非脚掌接触 | 负奖励 |
| flat_orientation | 惩罚非直立姿态 | 负奖励 |
| reach_goal | 到达目标点奖励（需 env.goal_positions 有效，默认 track 地形生效） | 正奖励 |

> 其它框架内置 reward（如 energy、feet_stumble、feet_height_body、termination 等）未默认激活，选手若需启用需在 `reward_process.py` 实现并在 TOML 里新增 `[rewards.<name>]` 段。

### 终止与超时协议

| 字段 | 类型 | Shape | 说明 |
|---|---|---|---|
| terminated | bool | (num_envs,) | 真实终止 |
| truncated | bool | (num_envs,) | 超时截断 |

| 条件 | terminated | truncated | 说明 |
|---|---|---|---|
| episode 最大时长 | False | True | 超时截断 |
| 姿态异常/摔倒 | True | False | 真实失败 |
| Track 到达出口 | True | False | 导航成功 |

> 算法侧应使用 `dones = terminated \| truncated` 判断环境是否结束。

### 访问示例

#### 解析 step() 返回

```python
def parse_env_step(data):
    if data is None:
        raise RuntimeError("env.step failed")
    frame_no, obs, rewards, terminated, truncated, extra = data
    infos, privileged_obs = extra
    critic_obs = privileged_obs if privileged_obs is not None else obs
    dones = torch.logical_or(terminated, truncated)
    return {
        "frame_no": frame_no,
        "obs": obs,
        "critic_obs": critic_obs,
        "rewards": rewards,
        "dones": dones,
        "terminated": terminated,
        "truncated": truncated,
        "infos": infos,
    }
```

#### 解析 policy observation

```python
def parse_policy_obs(obs: torch.Tensor):
    return {
        "base_ang_vel": obs[:, 0:3],
        "projected_gravity": obs[:, 3:6],
        "velocity_commands": obs[:, 6:9],
        "joint_pos_rel": obs[:, 9:21],
        "joint_vel_rel": obs[:, 21:33],
        "last_action": obs[:, 33:45],
        "height_scan": obs[:, 45:301],
    }
```

### 常见问题

**Q1: Critic 观测和 Policy 观测的区别？**

Critic 观测包含 critic_proprio（60 维），额外含 base_lin_vel 和 joint_effort 等特权信息，仅训练时可用，体现不对称 Actor-Critic 设计。

**Q2: 返回格式与传统环境的区别？**

reset() 返回 (obs, critic_obs) 而非字典。step() 返回 (frame_no, obs, rewards, terminated, truncated, (infos, privileged_obs))，需用 `dones = terminated | truncated` 合并。

---

## 腾讯开悟强化学习框架

### 综述

欢迎来到腾讯开悟！

腾讯开悟强化学习开发框架是基于强化学习系统系列技术标准打造的标准化开发套件。该框架为开发者提供了标准化的编程接口和丰富的工具集，支持开发者高效完成智能体开发、环境交互，以及模型的训练及预测流程。

#### 训练流程简介

本开发框架的完整训练流程如下图所示：

```
[开发任务描述]
     ↓
智能体-环境循环交互 → 样本处理 → 模型迭代优化 → 智能体模型更新
```

如图，完整训练流程包含以下关键环节：

| 环节 | 介绍 |
|---|---|
| **智能体-环境循环交互** | - 智能体将环境提供的观测和奖励处理为符合预测函数输入要求的数据；<br>- 调用预测函数，生成动作指令；<br>- 将智能体输出的动作指令处理为符合环境输入要求的数据；<br>- 环境执行动作后完成状态转移，并反馈新的观测数据和奖励数据； |
| **样本处理** | - 每个环境有不同的开始与结束逻辑，智能体与环境从开始到结束的完整交互过程，称为 episode；<br>- 智能体与环境每一次交互产生的结构化数据，称为样本；一个 episode 产生的样本序列称为轨迹；<br>- 对轨迹数据进行处理，转换为规范化训练样本(SampleData)； |
| **模型迭代优化** | - 基于训练样本，通过算法持续更新模型参数，实现策略优化； |
| **智能体模型更新** | - 智能体加载最新模型，与环境继续循环交互； |

该流程通过强化学习分布式计算框架提供的训练工作流实现。基于此，开发框架主要包含三大核心模块：

- **强化学习环境系统**：提供标准的强化学习环境接口。开发者可以通过标准接口，实现智能体与环境的交互。
- **强化学习智能体开发套件**：提供标准的强化学习智能体接口，以及算法库、模型组件库等工具函数库。开发者可以通过工具函数库快速完成智能体的构建。
- **强化学习分布式计算框架**：提供标准接口，支持开发者按需实现训练工作流，运行单机或分布式的训练及评估任务。

#### 代码包简介

开发者可以通过腾讯开悟平台所提供的强化学习项目使用开发框架。一个强化学习项目的代码目录如下：

```
📦 根目录
├── 📂 agent
│   ├── 📂 algorithm
│   │   └── 📄 __init__.py
│   │   └── 📄 algorithm.py
│   ├── 📂 conf
│   │   └── 📄 __init__.py
│   │   └── 📄 conf.py
│   │   └── 📄 train_env_conf.toml
│   ├── 📂 feature
│   │   └── 📄 __init__.py
│   │   └── 📄 definition.py
│   │   └── 📄 preprocessor.py
│   ├── 📂 model
│   │   └── 📄 __init__.py
│   │   └── 📄 model.py
│   ├── 📂 workflow
│   │   └── 📄 __init__.py
│   │   └── 📄 train_workflow.py
│   ├── 📄 __init__.py
│   └── 📄 agent.py
├── 📂 conf
│   ├── 📄 __init__.py
│   ├── 📄 configure_app.toml
├── 📂 log
└── 📄 train_test.py
```

代码目录介绍：

| 目录名 | 介绍 |
|---|---|
| `agent/` | 智能体子目录，智能体相关内容均集中于该目录，是开发者核心工作目录。 |
| `conf/` | 配置文件目录，包含运行训练任务相关的配置，例如训练样本批处理大小 batch_size 等。 |
| `log/` | 日志目录，存放运行代码测试脚本时生成的日志文件。 |
| `train_test.py` | 代码正确性测试脚本，该脚本会使用当前代码包完成一步训练。建议开发者在启动训练任务前，确保代码已通过该脚本检测。 |

##### agent

| 目录/文件名 | 介绍 |
|---|---|
| `algorithm/` | 算法相关，开发者在该目录下完成算法实现，包含 loss 计算、模型优化等，详情见[算法开发](#算法开发) |
| `feature/` | 特征相关，开发者在该目录下完成数据结构定义和数据处理方法，以及样本处理和奖励计算，详情见[特征处理](#特征处理) |
| `model/` | 模型相关，开发者在该目录下完成模型实现。详情见[模型开发](#模型开发) |
| `workflow/` | 工作流目录，开发者在该目录下完成训练工作流的开发。详情见[工作流开发](#工作流开发) |
| `agent.py` | 智能体核心代码文件，开发者在该文件中完成预测、训练等核心函数的实现。详情见[智能体开发](#智能体开发) |

> 标准代码包中都存在一个 `agent_diy` 子文件夹，该文件夹是预定义的智能体模板，可供开发者进行智能体的开发。

##### conf

| 文件名 | 介绍 |
|---|---|
| `configure_app.toml` | 训练任务相关的配置，包括样本大小、样本池大小等。 |

通过对训练流程和代码包的介绍，相信开发者能够对腾讯开悟开发框架建立了初步认知。

接下来，我们将详细介绍每个模块的功能及使用方式。

### 环境

在综述中提到，强化学习训练流程离不开智能体与环境的持续交互，本文将详细介绍强化学习环境系统的功能及标准接口函数。

#### 概述

强化学习环境是基于输入动作，输出观测、奖励等反馈的功能模块，用于表达强化学习算法所求解的问题场景。

开发框架通过场景适配模块，对仿真器进行封装，将其特化的接口、协议转换为强化学习环境统一的接口和协议，供智能体调用。

强化学习环境系统主要提供如下功能：

- 接收配置信息，用于指定自身初始化方式，比如环境中各种元素的初始状态。
- 输出观测、奖励信息，可用于智能体预测、训练。
- 输出观测、奖励之外的其他信息，供强化学习系统相关组件使用以实现特定功能。其他信息可包括可视化数据、日志数据等，实现的功能包括环境可视化、运行状况监测等。
- 接收动作指令，完成状态转移并产生新的观测和奖励。

#### 环境使用

开发框架通过场景适配模块，将问题场景进行标准化封装，为开发者提供统一的交互接口与通信协议。由于环境之间存在差异，接口中所涉及的观测、奖励等信息的具体数据结构也有所不同，开发者需查阅所使用环境的官方数据协议文档以获取准确信息。

开发者可以在训练工作流的 workflow 中获取到对应环境的实例，通过标准接口实现智能体与环境的交互。

#### 核心函数介绍

##### reset(usr_conf)

reset 会将环境重置为环境配置文件中指定的状态，并且返回初始观测。

```python
# usr_conf为开发者传入的环境配置
obs, state = env.reset(usr_conf=usr_conf)
```

**Parameters**

| 参数名 | 介绍 |
|---|---|
| usr_conf | dict 类型，环境配置文件 |

**Returns**

| 参数名 | 介绍 |
|---|---|
| obs | dict 类型，环境观测信息 |
| state | dict 类型，环境全局信息 |

##### step(act, stop_game = false)

环境会执行传入的 act 动作指令，完成一次状态转移，并返回新的观测和奖励等信息。

```python
frame_no, _obs, score, terminated, truncated, _state = env.step(act, stop_game=false)
```

**Parameters**

| 参数名 | 介绍 |
|---|---|
| act | dict 类型，环境执行的动作 |
| stop_game | bool 类型，是否结束当前对局 |

**Returns**

| 参数名 | 介绍 |
|---|---|
| frame_no | int 类型，当前环境实例运行时的帧号 |
| _obs | dict 字典类型，当前帧的观测信息 |
| score | int 类型，当前帧的奖励信息 |
| terminated | bool 类型，当前环境实例是否结束 |
| truncated | bool 类型，当前环境实例是否异常或中断 |
| _state | dict 字典类型，当前帧的全部状态信息 |

### 智能体

智能体是强化学习系统中的核心模块，在开发框架综述中提到，完整训练流程包括：

| 环节 | 介绍 |
|---|---|
| **智能体-环境循环交互** | - 智能体将环境提供的观测和奖励处理为符合预测函数输入要求的数据；<br>- 调用预测函数，生成动作指令；<br>- 将智能体输出的动作指令处理为符合环境输入要求的数据；<br>- 环境执行动作后完成状态转移，并反馈新的观测数据和奖励数据； |
| **样本处理** | - 每个环境有不同的开始与结束逻辑，智能体与环境从开始到结束的完整交互过程，称为 episode；<br>- 智能体与环境每一次交互产生的结构化数据，称为样本；一个 episode 产生的样本序列称为轨迹；<br>- 对轨迹数据进行处理，转换为规范化训练样本(SampleData)； |
| **模型迭代优化** | - 基于训练样本，通过算法持续更新模型参数，实现策略优化； |
| **智能体模型更新** | - 智能体加载最新模型，与环境继续循环交互； |

基于上述训练流程，我们将智能体的开发分为四个部分：

1. **数据处理及奖励设计**：介绍基于环境观测数据进行特征处理、样本处理和奖励设计的方法。
2. **模型开发**：介绍模型开发接口及开发方法。
3. **算法开发**：介绍包括算法开发接口及开发方法。
4. **工作流开发**：介绍开发者开发自定义训练工作流的方法。

接下来，将通过独立的章节对强化学习智能体开发套件中每个模块的功能及接口函数进行介绍。

### 特征处理

环境返回的数据通常无法直接作为智能体预测和训练的输入，开发者需要完成特征处理、样本处理和奖励设计，确保数据结构与类型符合智能体的接口规范。

#### 特征处理

在特征处理时，开发者需要完成四个关键的开发工作，分别是定义数据结构、观测处理、动作处理。

##### 定义数据结构

**开发目录**：`<智能体文件夹>/feature/definition.py`

首先，开发者需要定义智能体可以使用的数据结构（类）。

开发框架已经预先定义好了三种数据类型：ObsData, ActData, SampleData。

- ObsData 和 ActData 分别表示智能体预测的输入和输出，将会由 `agent.predict()` 使用；
- SampleData 为训练样本的数据类型，训练样本将会被 `agent.learn()` 使用，进行模型训练。

**核心函数介绍**

###### create_cls

用于动态创建数据结构（类）。ObsData, ActData, SampleData 是训练流程必需的三类，但每一个类的数据结构包含哪些属性完全由开发者自定义，属性名称和属性数量没有限制。

```python
ObsData = create_cls("ObsData", 
    feature=None, 
)
ActData = create_cls("ActData",
    action=None,
    prob=None,
)
SampleData = create_cls("SampleData",
    npdata=None
)
```

| 参数名 | 介绍 |
|---|---|
| 第一个参数 | 字符串类型，类的名称 |
| 其余参数 | 类的属性，默认值为 None，由开发者自行定义 |

##### 观测处理

**开发目录**：`<智能体文件夹>/agent.py`

由于环境的 reset 和 step 接口返回的数据属于原始观测数据，无法直接作为智能体预测时的输入，开发者需要将这部分数据进行特征化。

**核心函数介绍**

###### observation_process

将环境返回的观测数据转换成 ObsData 类型数据。

很多情况下，特征工程包含了大量的数值处理、数据转换和领域知识，我们建议将大量的特征处理代码在 `<智能体文件夹>/feature/preprocessor.py` 文件中实现，然后由于 observation_process 进行调用。

```python
def observation_process(self, obs, state=None):
    return ObsData(feature=feature, legal_act=legal_actions)
```

| 参数名 | 介绍 |
|---|---|
| obs | Observation 类型，env.reset 和 env.step 返回的环境观测数据 |
| state | EnvInfo 类型，env.reset 和 env.step 返回的环境状态数据 |

**Return**

| 参数名 | 介绍 |
|---|---|
| ObsData | 开发者定义的 ObsData 类型的数据，将作为 agent.predict() 函数的输入。 |

##### 动作处理

**开发目录**：`<智能体文件夹>/agent.py`

由于环境的 step 接口的输入须要满足环境的特定数据协议，开发者需要将智能体预测的输出转换为符合环境 step 接口输入要求的数据。

**核心函数介绍**

###### action_process

将智能体预测输出的 ActData 类型数据转换成环境可以接收的动作数据。

```python
def action_process(self, act_data):
    return act_data.act
```

| 参数名 | 介绍 |
|---|---|
| act_data | 开发者定义的 ActData 类型的数据 |

**Return**

环境能处理的动作数据类型，作为 env.step() 的输入

#### 奖励设计

**开发目录**：`<智能体名称>/feature/definition.py`

这里的奖励特指强化学习中的 Reward，注意要与环境反馈的 Score 进行区分。Score 通常用于衡量智能体在环境中的实际表现。开发者在设计 Reward 时，有非常大的灵活性，不仅可以基于环境返回的观测信息，还可以加入开发者对问题的理解、经验或者知识。

**核心函数介绍**

###### reward_shaping

开发框架预设的奖励设计函数接口，开发者可以通过该函数实现复杂的奖励计算，在训练工作流中调用。

```python
def reward_shaping(obs, _obs, state, _state):
    return reward
```

| 参数个数和类型不限制，可以是环境信息、智能体信息、开发者的经验和知识等。 |

**Return**

数值类型，计算出的 reward 值

#### 样本处理

**开发目录**：`<智能体文件夹>/feature/definition.py`

由于环境与智能体交互过程中产生的轨迹数据无法直接作为智能体训练时的输入，开发者需要将轨迹数据转换为训练样本数据。

**核心函数介绍**

###### sample_process

将环境与智能体交互过程中产生的轨迹数据转换成开发者定义的 SampleData 类型数据。

```python
@attached
def sample_process(self, list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]
```

| 参数名 | 介绍 |
|---|---|
| list_game_data | list(Frame) 类型， 使用开发者自定义的 Frame 作为输入，因为样本一般进行批处理，所以传入列表 |

**Return**

| 参数名 | 介绍 |
|---|---|
| list(SampleData) 类型 | SampleData 类型的数据组成的列表 |

为了支持分布式训练，样本数据需要进行网络传输，由于 SampleData 无法直接进行网络传输，需要先转换成 Numpy 的 Array，待传输到对端之后再由 np.Array 转换成 SampleData。

因此，开发者需要实现两个转换函数 SampleData2NumpyData 和 NumpyData2SampleData，这两个函数互为反函数。

> **注意**：由于这两个函数会被分布式计算框架调用，因此这两个函数的实现都必须包含一个装饰器 @attached

###### SampleData2NumpyData

将 SampleData 转换为 NumpyData。

```python
@attached
def SampleData2NumpyData(g_data):
    return g_data.npdata
```

| 参数名 | 介绍 |
|---|---|
| g_data | SampleData 类型 |

**Return**

Numpy.array 类型

###### NumpyData2SampleData

将 NumpyData 转换为 SampleData。

```python
@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)
```

| 参数名 | 介绍 |
|---|---|
| s_data | Numpy.array 类型 |

**Return type**

SampleData 类型

### 算法开发

**开发目录**：`<智能体名称>/algorithm/algorithm.py`

在完成特征处理和奖励设计后，开发者还需要实现强化学习算法，以通过特定优化方法更新模型参数。

以下为 实现强化学习算法的核心函数介绍，有关函数的更多细节可以查阅分布式计算框架

#### 核心函数介绍

##### learn

实现强化学习优化算法的核心方法，该函数输入为训练样本数据，开发者需基于不同的算法完成相关实现，包括优化方法、损失计算等。

```python
def learn(self, list_sample_data):
    """
    Implementing the core method of the algorithm
    实现算法的核心方法
    """
    loss = 0                         # 基于不同算法实现loss计算 Calculate loss
    loss.backward()                  # 计算梯度 Calculate gradient
    self.optimizer.step()            # 通过梯度下降等方法更新模型 Update weights 
```

| 参数名 | 介绍 |
|---|---|
| list_sample_data | list 类型，训练样本(SampleData)列表 |

### 模型开发

**开发目录**：`<智能体名称>/model/model.py`

一个强化学习模型是基于特征作为输入数据，输出策略的神经网络模型。

开发者需要在 model.py 文件中，实现神经网络模型。开发框架要求，模型类需继承 `torch.nn.Module` 类，即符合 Pytorch 模型的实现规范。

```python
class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
```

### 工作流开发

#### 训练工作流

在完成智能体开发后，需要进一步实现由分布式计算框架提供的训练工作流接口，使智能体和环境持续交互，收集训练样本，迭代模型参数，最终完成策略的优化。

#### 核心函数介绍

##### workflow

通过该函数实现强化学习训练工作流，调用智能体和环境提供的接口，完成环境交互、样本收集和模型更新。

```python
@attached
def workflow(envs, agents, logger=None, monitor=None):
```

| 参数名 | 介绍 |
|---|---|
| envs | list 类型，环境列表，返回当前正在运行的环境。 |
| agents | list 类型，智能体列表，通过调用开发者实现的 `<智能体名称>/agent.py` 实例化 Agent, 并作为输入传入 workflow。 |
| logger | Logger 类型，框架提供的日志组件，接口与 python 的 logging 库一致。 |
| monitor | Monitor 类型，框架提供的监控组件。 |

接下来，我们将通过一个训练工作流关键步骤的代码示例（具体实现由开发者完成），说明如何通过训练工作流实现完整训练流程。

```python
@attached
def workflow(envs, agents, logger=None, monitor=None):
    # Get the environment and agent
    # 获取环境和智能体
    env, agent = envs[0], agents[0]

    # Execute several epochs
    # 执行若干次epoch
    epoch_num = 1000
    
    # Each epoch executes several episodes
    # 每个epoch执行若干个episode
    episode_num_every_epoch = 1000

    # Training loop
    # 训练循环
    for epoch in range(epoch_num):
        # After each episode, the trajectory data is converted into training samples for training.
        # 在每一个episode结束之后，将轨迹数据转换成训练样本进行训练
        for g_data in run_episodes(episode_num_every_epoch, env, agent, logger, monitor):
            # Agent training. If single-machine training, the model is trained directly; if distributed training, samples are sent to the sample-pool.
            # agent进行训练。如果是单机训练，则直接对模型进行训练；如果是分布式训练，则将训练样本发送到样本池。
            agent.learn(g_data)
            # Ensure that the next training sample collected is new
            # 清空g_data，确保下一次搜集的训练样本是新的
            g_data.clear()
        
        # Save the model at intervals
        # 依据时间间隔保存模型
        now = time.time()
        if now - last_save_model_time >= 300:
            agent.save_model()
            last_save_model_time = now


def run_episodes(n_episode, env, agent, logger, monitor):
    # Run several episodes
    # 运行若干个episode
    for episode in range(n_episode):
        # Reset data at the beginning of an episode
        # 在episode开始时重置数据
        done = False
        collector = list()

        # Reset enviroment and get initial info
        # 重置环境, 并获取环境初始状态
        obs, state = env.reset(usr_conf=usr_conf)

        # Load the latest model and call it on demand; if in stand-alone mode, there is no need to load the remote model
        # 加载最新模型，按需调用；若训练采用单机模式，则无需加载远程模型，可不调用该函数
        agent.load_model(id="latest")

        # Run an episode loop
        # 运行一个episode循环
        while not done:
            # Agent performs inference, gets the predicted action for the next frame
            # 调用智能体预测函数，获取下一时刻的动作
            act_data = agent.predict(list_obs_data=[obs_data])[0]

            # Unpack ActData into action
            # 将智能体输出的ActData数据转换为符合环境数据协议要求的动作数据
            act = agent.action_process(act_data)

            # Interact with the environment, execute actions, get the next state
            # 调用环境step接口，与环境交互, 执行动作, 获取下一时刻的状态
            frame_no, _obs, score, terminated, truncated, _state = env.step(act)
            if _obs == None:
                break

            # Feature processing
            # 对环境返回的观测数据进行处理
            _obs_data = agent.observation_process(_obs, _state)

            # Disaster recovery
            # 容灾
            if truncated and frame_no == None:
                break

            # Calculate reward
            # 计算reward
            reward = reward_shaping(obs_data, _obs_data, state, _state)

            # Episode done signal
            # episode结束信号
            done = terminated or truncated

            # Construct sample
            # 构造样本
            frame = Frame(
                obs=obs_data.feature,
                _obs=_obs_data.feature,
                act=act,
                rew=reward,
                done=done,
            )
            collector.append(frame)

            # If the game is over, the sample is processed and sent to training
            # 如果episode结束，则进行样本处理，将样本送去训练
            if done:
                if len(collector) > 0:
                    collector = sample_process(collector)
                    # Return samples
                    # 返回样本数据, agent会调用agent.learn(g_data)进行训练
                    yield collector
                break

            # Status update
            # 状态更新
            obs_data = _obs_data
            obs = _obs
            state = _state
```

### 智能体开发

**开发目录**：`<智能体名称>/agent.py`

在完成模型和算法后，开发者还需要实现强化学习智能体，智能体使用模型进行决策、与环境交互并通过算法更新模型参数。

以下为 实现强化学习智能体的核心函数介绍，有关函数的更多细节可以查阅分布式计算框架

#### 核心函数介绍

##### learn

该函数输入为训练样本数据，开发者需要在该函数中调用算法消费训练样本进行训练。

当然，在不同的训练模式下，该函数使用方法有所不同：

- **单机训练**：开发者需要在训练工作流中手动调用该函数以进行一步训练。
- **分布式训练**：
  - 该函数作为训练函数会被循环执行，无需开发者手动调用。
  - 但该函数还作为样本发送函数，开发者需要在训练工作流中手动调用，以将样本发送至样本池。

```python
def learn(self, list_sample_data):
    self.algo.learn(list_sample_data)        # 调用算法消费训练样本进行训练 Call algorithm to train model
```

| 参数名 | 介绍 |
|---|---|
| list_sample_data | list 类型，训练样本(SampleData)列表 |

##### predict

该方法通过调用模型进行预测，通常在训练时调用该方法，依策略的概率分布采样或引入随机概率。

```python
@predict_wrapper
def predict(self, list_obs_data, list_state):
    return [ActData]
```

| 参数名 | 介绍 |
|---|---|
| list_obs_data | list 类型，观测数据(ObsData)列表 |
| list_state | 可选参数，list 类型，环境返回的状态数据列表 |

**Return**

| 参数名 | 介绍 |
|---|---|
| List(ActData) | list 类型，开发者定义的动作数据(ActData)列表 |

##### exploit

该方法通过调用模型进行预测，通常在评估时调用该方法，选取策略中概率最高的动作或者策略认为最优的动作。

```python
@exploit_wrapper
def exploit(self, observation):
```

| 参数名 | 介绍 |
|---|---|
| observation | dict 类型，环境观测字典，评估工作流中将原始的环境观测信息作为输入传入 agent.exploit()。 |

**Return**

| 参数名 | 介绍 |
|---|---|
| action | list 类型，动作列表，环境可以直接使用的动作指令 |

##### load_model

智能体通过该接口完成模型参数加载。在上文中提到，Actor 会从模型池中获取最新模型参数文件，开发者需要手动调用 load_model() 函数，使智能体完成模型参数加载。

```python
@load_model_wrapper
def load_model(self, path=None, id="1"):
    # When loading the model, you can load multiple files,
    # and it is important to ensure that each filename matches the one used during the save_model process.
    # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
    model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
    self.model.load_state_dict(
        torch.load(model_file_path, map_location=self.device),
    )
```

| 参数名 | 介绍 |
|---|---|
| path | string 类型，加载模型参数文件的路径，开发框架根据使用场景得到相应的路径, 并作为输入传入 load_model |
| id | string 类型，模型参数文件的 id，使用 id 指定加载的模型参数文件 |

##### save_model

开发者可以通过该函数保存当前时刻的模型文件及智能体代码包，开发框架会将开发者需要保存的内容打包为 zip 格式的文件。

当开发者使用腾讯开悟客户端开发时，开发框架会在客户端指定目录下存储该 zip 文件。
当开发者使用腾讯开悟平台时，开发框架会将该 zip 文件存储在云端，开发者可以通过平台的训练管理模块查看每一个训练任务的 zip 文件，即模型。

```python
@save_model_wrapper
def save_model(self, list_obs_data, list_state):
    # To save the model, it can consist of multiple files,
    # and it is important to ensure that each filename includes the "model.ckpt-id" field.
    # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
    model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

    # Copy the model's state dictionary to the CPU
    # 将模型的状态字典拷贝到CPU
    model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
    torch.save(model_state_dict_cpu, model_file_path)
```

| 参数名 | 介绍 |
|---|---|
| path | string 类型，模型文件保存的路径，开发框架根据使用场景得到相应的路径, 并作为输入传入 save_model |
| id | string 类型，模型文件的索引，开发框架获取到模型池中最新模型的索引, 并作为输入传入 save_model |

---

## 监控与日志

本文介绍如何使用腾讯开悟平台的监控与日志功能，帮助您实时掌握训练状态、快速定位问题。

### 监控

#### 查看监控面板

在训练管理页面，点击查看监控按钮，即可打开监控面板。

#### 监控面板组成

监控面板包含两个核心模块：

| 模块 | 功能 |
|---|---|
| 错误日志数量 | 展示各模块的错误日志统计，点击可查看详情 |
| 监控指标图 | 展示训练过程中的各类数据指标 |

#### 指标分类

监控指标分为四类：

| 分类 | 说明 |
|---|---|
| 基础指标（basic） | 训练进度相关的核心数据，如训练步数、预测次数等 |
| 硬件指标（hardware） | 资源使用情况，如 CPU、GPU、内存利用率 |
| 算法指标（algorithm） | 算法相关数据，不同算法的指标有所不同 |
| 环境指标（env） | 环境相关数据，不同环境的指标有所不同 |

#### 基础指标

| 面板名称 | 指标名称 | 说明 |
|---|---|---|
| 训练累计步数 | train_global_step | agent.learn() 的调用次数 |
| 预测累计次数 | predict_succ_cnt | agent.predict() 的调用次数 |
| 模型加载次数 | load_model_succ_cnt | agent.load_model() 成功调用的次数 |
| 样本接收次数 | sample_receive_cnt | 接收到的样本总数 |
| 已结束任务数 | episode_cnt | 已完成的 episode 数量 |
| 样本生产消耗比 | sample_production_and_consumption_ratio | 训练消耗样本数 / 采样生产样本数 |

#### 硬件指标

| 面板名称 | 指标名称 | 说明 |
|---|---|---|
| CPU 使用率 | aisrv_cpu_usage / learner_cpu_usage | 分别对应 aisrv 和 learner 进程 |
| GPU 使用率 | aisrv_gpu_usage / learner_gpu_usage | 分别对应 aisrv 和 learner 进程 |
| GPU 显存使用率 | gpu_memory | 显存占用百分比 |
| 内存使用率 | ram_usage | 容器内存占用，过高可能导致 OOM |

#### 算法指标

不同算法的指标各不相同，详见具体算法文档。

#### 环境指标

不同环境的指标各不相同，详见具体环境文档。

#### 自定义监控面板

平台支持在实验代码中自定义监控指标，配置并上报期望观测的业务数据。

##### 配置面板

在算法配置目录下编辑监控配置文件：

- **文件路径**：`agent_{算法名}/conf/monitor_builder.py`

配置示例：

```python
def build_monitor():
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("项目名称")
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(
            name="累积回报",
            description="反映智能体能力的指标",
            type="line",
        )
        .add_metric(metrics_name="reward", expr="avg(reward{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
```

##### 上报指标数据

在代码中调用监控上报接口，代码示例：

```python
import os
from monitor import monitor

monitor_data = {
    "reward": 100.5
}
monitor.push_data({os.getpid(): monitor_data})
```

##### 配置参数说明

| 配置项 | 字段 | 字段类型 | 说明 | 限制条件 |
|---|---|---|---|---|
| 项目名称 | title | string | 监控面板名称（该字段不在监控页面进行展示，不建议修改） | 支持中英文、数字、=、+、/、@、#、_、-及空格，长度 1~100 字符 |
| 面板组 | group_name | string | 面板组名称 | 支持中英文、数字、_、-及空格，长度 1~20 字符 |
| | group_name_en | string | 面板组英文标识 | 支持英文、数字、_、-及空格，长度 1~50 字符 |
| 面板 | name | string | 面板名称 | 支持中英文、数字、_、-及空格，长度 1~20 字符 |
| | name_en | string | 面板英文标识 | 支持英文、数字、_、-及空格，长度 0~50 字符 |
| | description | string | 面板描述信息 | 支持中英文、数字、标点符号等，长度 0~200 字符 |
| | type | string | 图表类型 | 仅 line（折线图）、stat（数值图）有效 |
| | unit | string | 数值单位，当 type 为 stat 时展示到指标后 | 仅 stat 类型有效 |
| 指标 | metrics_name | string | 指标显示名称 | 支持中英文、数字、_、-、{}及空格，长度 1~40 字符 |
| | expr | string | 指标查询表达式，支持指标变量（lable） | 使用 PromQL 的查询语法即可 |

规格限制：

- 折线图面板：指标数量限制为 20 个指标
- 数值图面板：指标数量限制为 2 个指标

##### 指标变量说明

如需按维度分组展示数据（如不同对手的胜率），可通过**指标变量（label）**区分维度，并在配置中使用指标变量定义展示方式。

**第一步：上报带 label 的数据**

```python
# 上报玩家1的胜率指标
monitor.push_data({os.getpid(): win_rate, "player": "player1"})

# 上报玩家2的胜率指标
monitor.push_data({os.getpid(): win_rate, "player": "player2"})
```

**第二步：配置指标变量**

```python
.add_metric(
    metrics_name="win_rate_{player}",
    expr="avg(win_rate{}) by (player)"
)
```

说明：

- `metrics_name` 字符串中的 `{player}` 会被实际的 player1、player2 替换；最终将在同一个面板上显示两条数据线：win_rate_player1 和 win_rate_player2

### 日志

框架提供了统一的日志服务，帮助您记录和排查训练过程中的问题。

#### 日志格式

| 字段 | 说明 | 示例 |
|---|---|---|
| time | 时间戳 | 2024-09-18 19:33:04.813469 |
| level | 日志级别 | INFO / WARNING / ERROR |
| message | 日志内容 | kaiwu learner train count is 365676 |
| file | 源码文件 | on_policy_trainer.py |
| line | 代码行号 | 769 |
| module | 所属模块 | learner |
| process | 进程名 | on_policy_trainer |
| function | 函数名 | train_stat |
| stack | 错误堆栈 | 仅错误日志包含 |

> **注意**：
> - **不要重写日志系统**：重写后监控面板将无法统计错误日志数量
> - **日志流量限制**：框架限制为 60 条/分钟，超出部分将被丢弃

---

## 强化学习系统系列技术标准

> 本文档所描述的腾讯开悟强化学习开发框架是基于强化学习系统系列技术标准打造的标准化开发套件。

---

*Copyright © 1998 - 2026 Tencent. All Rights Reserved.*
