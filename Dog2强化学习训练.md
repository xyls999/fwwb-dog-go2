# Dog2/Go2 强化学习训练项目说明

本文档用于记录当前 Dog2/Unitree Go2 四足机器人强化学习训练项目的完整上下文，包括比赛规则、项目结构、训练流程、算法路线、奖励函数设计、当前配置特点、调参经验和后续训练策略。

当前项目目标不是单纯让机器狗能走，而是在腾讯 AI Arena KaiwuDRL + Isaac Lab 仿真环境中，让 Go2 在复杂 Track 赛道上尽可能快、稳、省电地完成任务，最终冲击更高平均分。

更新时间：2026-05-23

---

## 1. 项目目标

### 1.1 任务定位

本项目对应四足机器人自主导航运控赛题。智能体需要控制 Unitree Go2 机器狗，在 Isaac Lab 仿真环境中通过强化学习学会：

- 在坡面、反向坡面、楼梯、反向楼梯、迷宫等地形上运动。
- 在未知或半未知地形中自主寻找前进路线。
- 在 Track 模式下从起点移动到终点。
- 同时兼顾速度、姿态稳定性、能耗效率和完成率。

项目当前重点是 Track 模式，也就是完整赛道模式。

### 1.2 你的训练目标

你的核心目标一直很明确：

- 不是只要完成率高，而是要把平均分推到 70、75，甚至 80+。
- 不能满足于快走，希望模型真正学会跑、冲刺、跳跃、借助地形前进。
- 希望策略在不同地形上能自动找到得分更高的路线，而不是永远走最吃力的中线。
- 希望速度、姿态、电量三者达到比赛评分意义上的最优平衡。
- 希望用已有预训练模型中的动作能力作为基础，再通过更贴近平台评分的奖励函数进行收束。

当前训练经验说明：完成率高不等于总分高。Track 总分外面乘完成系数，但完成之后还要看时间、姿态、电量。速度过高会提升时间分，但可能拉低姿态和能耗；约束过强又会让狗回到快走，学不到真正跑步。

---

## 2. 比赛评分逻辑

### 2.1 Track 模式总分

Track 模式核心评分公式可以概括为：

```text
total_score = completion_coeff * (0.4 * time_score + 0.4 * pose_score + 0.2 * energy_score)
```

其中：

- `completion_coeff`：完成系数，完成赛道的机器狗数量 / 总机器狗数量。
- `time_score`：时间分，用时越短越高。
- `pose_score`：姿态分，roll/pitch 偏移越小越高。
- `energy_score`：能耗分，平均机械功率越低越高。

这意味着 Track 模式的真实优先级是：

1. 先完成，没完成时其他分数意义很小。
2. 完成后，时间和姿态同样重要，各占 40%。
3. 能耗占 20%，但如果能耗太差，也会明显拖低总分。

### 2.2 姿态分

姿态分近似为：

```text
pose_score = 100 * exp(-5 * mean(|roll| + |pitch|))
```

所以姿态分对 roll/pitch 很敏感。高速、跳跃、下楼、急转弯都会显著影响姿态分。

关键经验：

- 外八或内八本身不是绝对好坏，关键看它是否导致 roll/pitch 抖动、足端打滑、关节高功率。
- 上楼外八、下楼内八如果能稳定通过，可能对完成有帮助；如果导致身体左右摆、脚横向刮地，则会同时伤姿态和电量。
- 真正高分的跑法不是乱跳，而是高速下仍保持躯干平稳。

### 2.3 能耗分

能耗分近似为：

```text
energy_score = 100 * exp(-0.01 * mean(sum(abs(joint_vel) * abs(applied_torque))))
```

能耗看的是关节速度和关节力矩乘积。高频抖腿、大幅摆腿、贴墙摩擦、楼梯绊脚都会拉低能耗分。

关键经验：

- 高速度通常会提升时间分，但会带来更高关节速度和更大扭矩。
- 不能只加速度奖励，否则模型会用很贵的动作换时间分。
- 能耗奖励太强又会让模型保守，甚至变成慢走或不愿冲刺。

### 2.4 时间分

时间分要求尽快完成。它和速度强相关，但不是简单的速度越大越好：

- 如果 5m/s 命令导致姿态差、撞墙、卡台阶，完成率和姿态会掉。
- 如果速度命令不稳定，模型会学到杂乱步态，表现为忽快忽慢。
- 如果速度发布太高但奖励没有给足跑步结构，模型可能只是用快走硬追速度，无法形成跑步步态。

---

## 3. 仓库结构

项目主要目录如下：

```text
agent_ppo/
  conf/                 PPO 阶段配置与训练 TOML
  feature/              观测处理、奖励函数、导航辅助、目标点处理
  model/                Actor-Critic 网络
  algorithm/            PPO 算法实现
  workflow/             训练循环与速度命令覆盖逻辑
  agent.py              KaiwuDRL 智能体入口

agent_diy/
  conf/                 DIY 混合 CPG + RL 配置
  feature/              DIY 观测与奖励
  model/                残差策略网络
  algorithm/            PPO 风格训练逻辑
  workflow/             CPG + RL residual + reflex 训练流程

isaac_env/
  base_env.py           Isaac Lab 环境包装、TOML 合并、reward 注册、step/reset

conf/
  app/algo/replay/model dump 等 KaiwuDRL 全局配置

ckpt/
  本地 checkpoint 与元信息

introduce.md
  官方比赛介绍、评分、环境、监控指标、开发流程

秘密武器.md
  更详细的评分理解、策略经验、训练技巧

奖励函数设定解析.md
  奖励函数解释和调参思路

reward_alignment_report_1.md
  当前奖励函数与评分标准对齐报告

evaluate_pareto.py
  离线筛选 checkpoint，用 forward/time/pose/energy 做 Pareto 排序

train_test.py
  训练冒烟测试入口
```

---

## 4. 当前主线：agent_ppo

### 4.1 为什么当前主线用 PPO

当前项目主线仍然是 PPO，而不是换成别的强化学习算法。原因：

- 代码包、平台接口、模型保存和训练流程都围绕 PPO 写好。
- PPO 对四足机器人连续控制任务非常常见，稳定性高。
- 当前瓶颈主要不是 PPO 算法本身，而是奖励函数、速度命令、探索强度、地形采样和预训练模型收束方式。
- 换算法会带来接口、采样、存储、评估、checkpoint 兼容等大量风险。

当前策略是保留 PPO，把核心精力放在：

- 奖励函数是否贴近平台评分。
- 速度目标是否合理。
- 是否能诱导跑步而不是快走。
- 是否能在完成率、姿态、能耗之间找到最优点。

### 4.2 PPO 模型结构

当前 PPO 主线配置位于：

```text
agent_ppo/conf/conf.py
```

当前 TrackNavConfig：

```python
name = "nav"
task_type = "track"
num_actions = 12
num_proprio_obs = 45
num_scan = 256
num_goal_obs = 3
num_critic_observations = 319
actor_hidden_dims = [512, 256, 128]
critic_hidden_dims = [512, 256, 128]
activation = "elu"
```

观测维度：

- Actor obs：45 + 256 + 3 = 304。
- Critic obs：316 + 3 = 319。
- Action：12 维，直接控制 Go2 的 12 个关节动作。

Actor 输入包括：

- 本体感知：关节位置、关节速度、机体角速度、重力方向、命令等。
- 高度扫描：16x16 height scan，共 256 维。
- 目标信息：机器人坐标系下 goal x/y + goal distance。

Critic 使用不对称 Actor-Critic：

- Critic 可以看到特权信息，例如 base linear velocity、joint effort。
- Actor 推理时不依赖这些特权量。

### 4.3 当前 PPO 超参数

当前 TrackNavConfig 是偏激进但仍可训练的版本：

```python
lr = 1.0e-4
num_learning_epochs = 5
num_mini_batches = 4
num_steps_per_env = 48
clip_param = 0.30
entropy_coef = 0.0060
desired_kl = 0.018
init_noise_std = 1.15
min_normalized_std = [0.10, 0.05, 0.10] * 4
max_normalized_std = [0.85, 0.55, 0.85] * 4
model_save_interval = 20
```

这些参数的含义：

- `lr=1e-4`：用于 fine-tune，避免高分模型被迅速破坏。
- `clip_param=0.30`：比常规 0.2 更激进，允许策略更新幅度更大。
- `entropy_coef=0.006`：保留一定探索，但不至于完全发散。
- `init_noise_std=1.15`：初始动作噪声偏大，用于探索跑跳动作。
- `max_normalized_std` 限制动作噪声上界，避免完全无序。

---

## 5. 备线：agent_diy

`agent_diy` 是另一条混合控制路线，不是当前主线。

它的特点是：

- CPG 生成基础周期步态。
- RL 输出 residual action，对 CPG 进行修正。
- Reflex 作为安全反射，用于倾倒恢复或异常状态处理。
- 模型比 PPO 主线更轻，因为 CPG 承担了部分运动先验。

DIY 的优势：

- 更容易形成周期性步态。
- 对稳定行走、低能耗可能有帮助。
- 更像“手工运动先验 + RL 微调”。

DIY 的风险：

- 动作空间被 CPG 先验限制，可能不利于学到非常激进的跑跳。
- Track 复杂导航和迷宫策略未必比端到端 PPO 更强。
- 当前调参和对齐主要集中在 `agent_ppo`，DIY 需要额外迁移成本。

当前结论：

- 冲击极限探索、跑跳、全地形高速：优先 PPO 主线。
- 如果后续需要稳定节能步态或姿态修复，可以借鉴 DIY 的 CPG/Reflex 思路。

---

## 6. 环境与地形配置

当前 Track 训练配置位于：

```text
agent_ppo/conf/train_env_conf_track_nav.toml
```

### 6.1 并行环境

当前：

```toml
[env]
num_envs = 4096
episode_length_s = 120.0
```

注意：

- 平台校验限制 `num_envs` 最大为 4096。
- 曾尝试设置 10000，但平台报错：允许范围是 `[1, 4096]`。
- 所以当前最大并行数就是 4096。

### 6.2 Track 地形

当前 Track：

```toml
[terrain.track]
track_length = 5
sub_terrains = [
  "pyramid_slope",
  "pyramid_slope_inv",
  "pyramid_stairs",
  "pyramid_stairs_inv",
  "open_entry_maze"
]
```

含义：

1. 正坡。
2. 反坡。
3. 上楼梯。
4. 下楼梯。
5. 开口迷宫。

这是完整 Track 任务，而不是单一 locomotion。

### 6.3 难度采样

当前 level mix：

```toml
[terrain.level_mix]
enabled = true
levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
weights = [0.14, 0.13, 0.11, 0.10, 0.10, 0.09, 0.09, 0.08, 0.08, 0.08]
pool_size = 100
```

这是偏均衡、低难略多的配置。之前尝试过加高中高难度权重，但出现过 L9 样本过多、得分被高难拖低的问题。

经验：

- 只训低难：短期分数好看，但高难泛化不足。
- 只训高难：学习噪声大，总分可能下降。
- 当前更适合先稳定完成，再逐步针对 L6-L9 加权。

### 6.4 域随机化

当前：

```toml
[domain_rand]
enable_domain_rand = true
randomize_friction = true
friction_range = [0.45, 1.4]
push_robots = false
```

含义：

- 摩擦随机化开启，提升泛化。
- 外力 push 关闭，避免高速跑跳阶段训练太乱。

---

## 7. 速度命令机制

### 7.1 官方 commands.limit 的作用

配置里仍然有：

```toml
[commands.limit]
lin_vel_x = [0.0, 1.0]
lin_vel_y = [-0.0, 0.0]
ang_vel_z = [-1.2, 1.2]
```

这个是 Isaac Lab 原生命令采样器的限制范围，主要用于官方 command manager 采样。

但是当前 `agent_ppo` 的 TrackNav 不是完全依赖官方原生命令采样。当前工作流里有 RL navigation phase command override，会手动覆盖：

- policy obs 里的 command 槽位。
- Isaac Lab command_manager 里的 `base_velocity`。

所以训练中真正让模型看到并追踪的速度锚点来自 `[rl_navigation]`。

### 7.2 当前速度配置

当前是激进探索版本：

```toml
[rl_navigation]
phase_command_enabled = true
pre_maze_lin_vel_x = [4.95, 5.05]
slope_lin_vel_x = [4.95, 5.05]
stairs_lin_vel_x = [4.95, 5.05]
maze_lin_vel_x = [4.95, 5.05]
phase_command_resample_steps = 64
terrain_phase_speed_enabled = true
eval_command_override = true
eval_command = [5.00, 0.0, 0.0]
```

含义：

- 平地/坡/楼梯/迷宫都给 5m/s 左右速度锚点。
- 范围很窄 `[4.95, 5.05]`，减少速度随机性干扰。
- 每 64 step 重采样一次，保证命令稳定但仍能更新。
- 评估时也覆盖为 5m/s 前进锚点。

注意：

- 这不是保分速度，是逼模型探索跑步/冲刺/跳跃的速度。
- 如果模型只是快走，说明奖励仍未形成跑步结构，或者动作空间探索还没打开。
- 如果分数下降，通常是姿态、电量和完成稳定性被速度目标打崩。

---

## 8. 当前奖励函数系统

奖励函数实现在：

```text
agent_ppo/feature/reward_process.py
```

奖励权重配置在：

```text
agent_ppo/conf/train_env_conf_track_nav.toml
```

当前奖励函数分为几类。

### 8.1 速度跟踪与冲刺奖励

主要目标：让模型追 5m/s，并从快走变成跑步。

关键项：

```toml
[rewards.track_lin_vel_xy]
weight = 8.00
std = 1.10

[rewards.command_speed_advantage]
weight = 5.00

[rewards.running_speed_band]
weight = 9.00
min_speed = 2.50
target_speed = 5.00
max_speed = 5.80

[rewards.forward_heading_velocity]
weight = 10.0
target_speed = 5.00

[rewards.goal_velocity_projection]
weight = 18.0
max_speed = 5.00
```

作用：

- `track_lin_vel_xy`：奖励实际速度跟随 command。
- `command_speed_advantage`：鼓励不要落后命令，惩罚慢于命令。
- `running_speed_band`：把速度推到跑步区间。
- `forward_heading_velocity`：奖励身体朝前高速移动。
- `goal_velocity_projection`：奖励朝目标方向的速度投影。

风险：

- 权重过大时，模型可能用高能耗动作硬追速度。
- 姿态约束太弱时，跑跳会导致 pose 下降。
- 如果跑步结构奖励不足，模型仍会快走。

### 8.2 跑步结构奖励

为了让模型不是简单快走，而是学会更大的步幅和滞空：

```toml
[rewards.feet_air_time]
weight = 1.20
threshold = 0.30

[rewards.feet_clearance]
weight = 0.70
target_height = 0.20

[rewards.feet_swing_forward]
weight = 1.20
target_forward = 0.45

[rewards.running_stride_span]
weight = 2.50
target_span = 0.95

[rewards.running_contact_pattern]
weight = 2.00
target_contacts = 2.0

[rewards.running_air_time_band]
weight = 2.00
target_air_time = 0.40
```

作用：

- `feet_air_time`：鼓励脚有足够滞空时间。
- `feet_clearance`：鼓励抬脚，减少绊楼梯。
- `feet_swing_forward`：鼓励摆腿向前，而不是原地高频抖。
- `running_stride_span`：鼓励前后脚跨度变大。
- `running_contact_pattern`：偏向跑步/小跑接触模式。
- `running_air_time_band`：鼓励合理滞空，但避免飞太久。

经验：

- 这些奖励是让快走变跑步的关键。
- 如果只加速度跟踪，不加步态结构，模型经常会用快走追速度。
- 如果滞空奖励过高，可能变成跳跃不稳，姿态分下降。

### 8.3 完成率与目标到达奖励

当前大幅增强了完成与目标推进：

```toml
[rewards.approach_goal]
weight = 60.0

[rewards.reach_goal]
weight = 220.0
threshold = 0.6

[rewards.task_complete]
weight = 1000.0
threshold = 0.6

[rewards.non_completion_timeout]
weight = -400.0

[rewards.navigation_time]
weight = -0.35

[rewards.survival]
weight = 0.30
```

作用：

- `approach_goal`：稠密奖励，鼓励持续接近目标。
- `reach_goal`：到达终点附近奖励。
- `task_complete`：完成完整 Track 的高额奖励。
- `non_completion_timeout`：超时未完成强惩罚。
- `navigation_time`：每步时间惩罚，逼模型更快。
- `survival`：存活奖励，避免激进探索中太快摔死。

当前这一组非常激进，目的是让模型优先冲向终点。

### 8.4 平台评分拟合奖励

当前保留了与评分公式直接对齐的稠密项：

```toml
[rewards.pose_score_formula]
weight = 0.80

[rewards.energy_score_formula]
weight = 0.20

[rewards.score_guidance]
weight = 0.15

[rewards.track_score_balance]
weight = 1.00
```

含义：

- `pose_score_formula`：近似平台姿态分 `exp(-5*(|roll|+|pitch|))`。
- `energy_score_formula`：近似平台能耗分 `exp(-0.01*power)`。
- `score_guidance`：速度跟踪、姿态、能耗的综合提示。
- `track_score_balance`：近似 Track 总分中的 0.4 time + 0.4 pose + 0.2 energy。

经验：

- 完全用评分函数做 reward 会很稀疏，且容易导致已经会完成的模型反而退化。
- 评分拟合项适合做收束，不适合从零开始承担所有探索。
- 当前版本为了跑步探索，把速度和完成奖励放得更强，姿态/能耗只是保留一定约束。

### 8.5 迷宫与墙体奖励

迷宫相关：

```toml
[rewards.maze_anticipatory_turn]
weight = 2.50

[rewards.wall_collision]
weight = -5.0

[rewards.wall_stall_penalty]
weight = -1.4

[rewards.wall_proximity]
weight = -0.05

[rewards.open_space]
weight = 1.00

[rewards.corridor_centering]
weight = -0.18

[rewards.directed_exploration]
weight = 0.035

[rewards.stuck_penalty]
weight = -2.2
```

作用：

- 提前转弯。
- 防撞墙。
- 防贴墙卡住。
- 鼓励走开阔区域。
- 在走廊中保持一定居中。
- 鼓励朝目标方向探索。

经验：

- 高速迷宫特别容易撞墙，墙体奖励不能完全去掉。
- 但是墙体感知也可能误把楼梯/坡当墙，所以很多墙体奖励通过 maze gate 只在接近迷宫时启用。

### 8.6 已经清零的保守限制项

为了不限制加速跑跳，当前清零了一批约束：

```toml
[rewards.lin_vel_z]
weight = 0.0

[rewards.joint_acc]
weight = 0.0

[rewards.correct_base_height]
weight = 0.0

[rewards.action_rate]
weight = 0.0

[rewards.action_smoothness]
weight = 0.0

[rewards.flat_orientation]
weight = 0.0

[rewards.posture_stability]
weight = 0.0

[rewards.hip_to_default]
weight = 0.0

[rewards.joint_position_penalty]
weight = 0.0

[rewards.dof_vel]
weight = 0.0

[rewards.air_time_variance_penalty]
weight = 0.0
```

目的：

- 放开跳跃和上下速度。
- 放开关节动作变化。
- 不强制固定身体高度。
- 不把策略锁死在慢走姿态。

风险：

- 姿态分会下降。
- 能耗会下降。
- 策略可能出现高频动作、乱跳、落地不稳。

当前这是探索阶段的主动取舍。

---

## 9. 训练流程

### 9.1 基本训练入口

冒烟测试：

```bash
python train_test.py
```

真实平台训练通常由 KaiwuDRL 框架调用：

1. 加载 `agent_ppo.agent.Agent`。
2. `Config.CURRENT = TrackNavConfig`。
3. 读取 `agent_ppo/conf/train_env_conf_track_nav.toml`。
4. 校验用户配置。
5. 创建 Isaac Lab 环境。
6. 注册 TOML 中的奖励函数。
7. 创建 PPO Actor-Critic。
8. 初始化 rollout storage。
9. 循环采样、更新 PPO、保存模型、上报监控。

### 9.2 训练循环

主训练循环位于：

```text
agent_ppo/workflow/train_workflow.py
```

核心流程：

```text
obs, critic_obs = env.reset()

for episode:
    for step in num_steps_per_env:
        obs = apply_rl_phase_command(obs)
        action = policy(obs)
        next_obs, reward, done, info = env.step(action)
        storage.add_transition(...)

    compute_returns()
    PPO.update()
    save_model_if_needed()
    report_monitor_data()
```

重点是 `apply_rl_phase_command`：

- 根据当前地形阶段或 goal distance 选择速度命令。
- 把 command 写入 obs 的 `[6:9]`。
- 同时尽量覆盖 Isaac Lab command_manager 的 `base_velocity`。
- 这样 reward 里的速度跟踪和 policy 看到的命令是一致的。

### 9.3 评估与视频

评估时重点看：

- `total_score`
- `time_score`
- `pose_score`
- `energy_score`
- `completed_count`
- `abnormal_count`
- `timeout_count`
- 各难度 `track_level_0/3/6/9` 的分数
- 视频中的步态是否真正在跑

视频判断非常重要，因为曲线可能显示速度提升，但模型实际上仍然是快走。

观察重点：

- 是否有明显滞空。
- 是否步幅变大。
- 是否身体上下跳太多。
- 是否脚横向刮地。
- 是否贴墙或走最高阻力路线。
- 上下楼是否绊脚。
- 迷宫是否提前转弯。

### 9.4 离线 checkpoint 排序

可以用：

```bash
python evaluate_pareto.py scores.jsonl --forward-threshold 90 --top-k 10 --plot
python evaluate_pareto.py scores.csv --format csv --pareto-front
```

用于从多个 checkpoint 中筛选：

- 完成率够高。
- time/pose/energy Pareto 更优。
- 不是只看单一 total score。

---

## 10. 我们已经验证过的经验

### 10.1 完成率高但得分下降

出现过这种情况：

- 旧模型完成率极高。
- 改成更贴近评分函数的 reward 后，总分反而下降。

原因：

- 已有模型已经形成可完成策略。
- 新 reward 改变了梯度方向，可能破坏原本的完成行为。
- 评分函数本身偏 episode 统计，直接做 step reward 会变稀疏或延迟。
- 姿态和能耗奖励太强会压制速度。
- 时间奖励太强又会破坏姿态和能耗。

结论：

- 评分拟合 reward 适合后期小学习率收束。
- 不适合突然大权重替换全部训练目标。

### 10.2 只加速度不一定会跑

模型不跑，只快走，可能原因：

- 高速命令只有目标，没有给跑步结构。
- action std 不够，探索不到跑跳动作。
- 姿态/能耗/平滑奖励太强，把跑跳压回快走。
- 预训练模型本身是走路策略，PPO fine-tune 容易围绕原步态局部优化。
- 5m/s 命令过高，模型追不上，反而学到不稳定动作。

解决方向：

- 增强足端滞空、步幅、前摆、接触模式奖励。
- 降低过强平滑/限高/姿态保守项。
- 提高探索噪声。
- 使用阶段训练：先探索跑，再收姿态/能耗。

### 10.3 靠边走奖励暂时失败

曾尝试让非 maze 地形靠边走更平路线，但效果不明显。

可能原因：

- height scan 对“边上更平”感知不可靠。
- 奖励无法明确区分坡、墙、台阶和可走边线。
- 机器人出生点、赛道坐标和局部 scan 坐标不一定直接对应。
- 靠边奖励和前进速度奖励冲突。

当前结论：

- 已先把非 maze 靠边奖励清零。
- 后续如果要恢复，应该先做可靠的地形/坐标判定，而不是直接用模糊 scan 奖励。

### 10.4 速度随机范围会干扰

之前速度范围较宽时，表现出速度不稳定。

当前改成：

```toml
[4.95, 5.05]
```

目的：

- 让模型学习稳定目标速度。
- 减少“今天追 3.2、明天追 4.7”的噪声。
- 让步态更容易收敛。

### 10.5 高难度采样不能一次拉太满

曾经提高高难度权重，但 L9 太多会拖垮平均分。

经验：

- 当低难和中难还没稳定高分，不宜把 L9 权重拉太高。
- 高难地形适合在已有稳定策略基础上做专项 fine-tune。
- 平均分目标是 75-80 时，先保证 L0/L3/L6 高分，再逐渐修 L9。

---

## 11. 当前版本特点

当前代码处于“激进跑步探索版”，不是最终保分版。

### 11.1 当前版本追求

- 强制全地形高速。
- 打开动作空间。
- 让模型尝试跑、跳、冲刺。
- 通过大完成奖励和时间惩罚避免拖慢。
- 通过存活奖励防止探索太快死亡。

### 11.2 当前版本不追求

- 当前不优先追求能耗最优。
- 当前不优先追求姿态满分。
- 当前不优先追求立刻平均分上涨。
- 当前不强制平滑动作或固定身体高度。

### 11.3 当前版本适合观察什么

训练后重点看：

- 是否终于从快走变成跑步。
- 是否出现明显更大步幅。
- 是否有合理滞空。
- 是否能在坡和楼梯保持前进。
- 是否仍然完成 Track。
- 是否 abnormal 大幅上升。
- 是否 pose 分掉到无法接受。
- 是否 energy 分过低。

---

## 12. 后续推荐训练策略

### 12.1 两阶段路线

建议把训练分成两个阶段。

第一阶段：动作探索阶段。

目标：

- 学会跑。
- 学会大步幅。
- 学会高速跨坡、跨楼梯。
- 不要求分数立刻上涨。

配置特点：

- 高速度命令。
- 高速度追踪奖励。
- 高跑步结构奖励。
- 低姿态/能耗约束。
- 中高探索噪声。

第二阶段：评分收束阶段。

目标：

- 保留跑步能力。
- 把姿态拉回来。
- 把能耗拉回来。
- 维持完成率。

配置特点：

- 速度从 5m/s 回收到实际得分最优区间。
- 增加 `pose_score_formula`。
- 增加 `energy_score_formula`。
- 恢复部分 `ang_vel_xy`、`joint_torques`、`feet_slide`。
- 降低 entropy 和 action std。
- 学习率降低。

### 12.2 可能的最终高分速度区间

从评分结构推断，最终最优速度不一定是 5m/s。

可能路线：

- 平地/下坡：更快。
- 上坡：略快但不能爆姿态。
- 楼梯：中高速，不能盲目 5m/s。
- 迷宫：能提前转弯时快，接近墙时慢。

最终可能会落在：

```text
平地/下坡：2.5 - 4.0+
上坡：2.2 - 3.5
楼梯：1.6 - 2.8
迷宫：1.8 - 3.0
```

当前 5m/s 更像探索上限，不一定是最终比赛速度。

### 12.3 平均分 75-80 的关键

要到 75-80，不是单点调参，而是同时满足：

- 完成系数接近 1。
- time_score 明显高，至少不能低于当前高分模型。
- pose_score 尽量 70+，最好 75-85。
- energy_score 不能长期低于 40，最好能回到 50+。
- L0/L3/L6 要稳定高分。
- L9 可以低一些，但不能完全拖垮。

高分模型应该像这样：

```text
completion_coeff: 0.95 - 1.00
time_score:       75+
pose_score:       75+
energy_score:     50+
total_score:      70 - 80+
```

### 12.4 回收策略

如果当前激进版本训练后：

- 会跑，但姿态差：提高 `pose_score_formula`，恢复一点 `ang_vel_xy`，降低 `lin_vel_z` 限制但不要太强。
- 会跑，但能耗差：提高 `energy_score_formula`，恢复轻微 `joint_torques` 和 `feet_slide`，降低无效大摆腿。
- 不会跑，只快走：继续强化 stride/air-time/contact pattern，保持高探索。
- 经常摔：恢复轻微姿态约束，降低楼梯/迷宫速度。
- 迷宫撞墙：提高 `wall_collision` 和 `maze_anticipatory_turn`，迷宫速度回收。
- 分数一路降：说明探索破坏了旧策略，应从上一个高分 checkpoint 重新开一条保守收束线。

---

## 13. 重要文件索引

### PPO 主线

```text
agent_ppo/conf/conf.py
```

控制：

- 当前训练阶段。
- Actor/Critic 网络维度。
- PPO 超参数。
- obs/action 维度。

```text
agent_ppo/conf/train_env_conf_track_nav.toml
```

控制：

- 并行环境数量。
- Track 地形。
- 地形难度采样。
- 速度命令。
- 奖励函数权重。
- 域随机化。

```text
agent_ppo/feature/reward_process.py
```

实现：

- 所有 `_reward_<name>()` 奖励函数。
- 平台评分拟合项。
- 跑步结构奖励。
- 导航与墙体奖励。

```text
agent_ppo/workflow/train_workflow.py
```

实现：

- PPO 训练循环。
- rollout 采样。
- phase command 覆盖。
- 监控上报。
- 速度课程/速度追踪统计。

```text
agent_ppo/agent.py
```

实现：

- Agent 初始化。
- 配置加载和校验。
- 模型创建。
- 推理时 command obs 覆盖。
- checkpoint 加载兼容。

### 环境

```text
isaac_env/base_env.py
```

实现：

- Isaac Lab 环境创建。
- TOML 合并到 env cfg。
- reward 注册。
- step/reset 包装。
- 评估监控信息整理。

### 文档与分析

```text
introduce.md
秘密武器.md
奖励函数设定解析.md
reward_alignment_report_1.md
```

用于理解：

- 官方规则。
- 评分公式。
- 训练流程。
- 奖励设计经验。
- 当前奖励与评分对齐程度。

---

## 14. 当前项目的核心特点

### 14.1 不对称 Actor-Critic

Actor 只看推理时能用的信息，Critic 在训练时看更多特权信息。这样可以提升训练效率，同时保持评估可用。

### 14.2 端到端关节控制

当前 PPO 主线直接输出 12 维关节动作，而不是输出速度给底层控制器。这给了模型更大自由度，也让跑跳成为可能。

### 14.3 手动速度锚点

当前不是完全信任官方 command sampler，而是在 workflow 中覆盖速度命令，让 policy 和 reward 都看到一致的速度目标。

### 14.4 奖励函数高度定制

当前奖励不仅有常规 locomotion，还包含：

- 平台评分近似。
- 跑步步态结构。
- Track 目标推进。
- 迷宫墙体处理。
- 完成/超时强约束。
- 探索阶段的存活奖励。

### 14.5 训练目标动态变化

项目不是一次性固定 reward，而是根据阶段切换：

- 探索阶段：跑起来。
- 完成阶段：走完整 Track。
- 收束阶段：对齐评分。
- 专项阶段：修姿态、电量、迷宫或高难。

---

## 15. 总结

这个项目的难点不在“让 Go2 能走”，而在于让它在完整 Track 上找到比赛分数意义下的最优运动方式。

当前最关键的矛盾是：

```text
速度越快，time_score 越可能提高；
但速度越快，pose_score 和 energy_score 越容易下降；
如果约束太强，模型又学不会跑，只会快走。
```

因此当前训练策略是：

1. 先用激进配置打开动作空间，逼模型探索跑步、冲刺、跳跃。
2. 确认视频中真的出现跑步模式。
3. 再从能跑的 checkpoint 出发，逐步收姿态和能耗。
4. 最后用平台评分拟合奖励做小学习率收束。

最终高分不是某一个奖励项带来的，而是完成率、时间、姿态、能耗、地形采样、速度发布、探索噪声共同平衡的结果。

