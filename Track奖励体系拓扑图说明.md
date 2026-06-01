# Track 奖励体系分类拓扑

本文档用于描述当前 Track 导航训练中的奖励函数体系，适合进一步绘制为结构图、流程图或知识图谱。整体奖励目标可以概括为：在保证机器人稳定姿态和合理能耗的前提下，提高 Track 赛道完成率、目标推进效率和复杂地形通过能力。

## 1. 总体拓扑结构

```text
Track 总奖励
├── A. 速度跟随与前进效率
├── B. 姿态稳定与官方姿态评分拟合
├── C. 能耗控制与官方能量评分拟合
├── D. 关节动作平滑与机械安全
├── E. 足端步态质量与复杂地形通过
├── F. 目标导航与完成任务
├── G. 迷宫避障与空间探索
└── H. 失败、超时与卡死抑制
```

## 2. A. 速度跟随与前进效率

该类奖励负责让机器人响应速度命令，并在赛道中保持有效前进。当前策略并不是无约束追求高速，而是在速度、姿态和能耗之间做平衡。

```text
A. 速度跟随与前进效率
├── track_lin_vel_xy              正向奖励，跟随环境速度命令
├── command_speed_advantage       正/负混合，低于命令速度惩罚，略高于命令速度奖励
├── goal_velocity_projection      正向奖励，鼓励速度投影朝向目标
└── backward_penalty              负向惩罚，抑制向后退
```

当前关键权重：

```text
track_lin_vel_xy:          0.88
command_speed_advantage:   0.60
goal_velocity_projection:  3.20
backward_penalty:         -0.80
```

图中建议表达为：速度命令跟随和目标方向推进共同驱动机器人前进，但超速奖励已被压低，避免速度目标压制姿态和能耗目标。

## 3. B. 姿态稳定与官方姿态评分拟合

该类奖励用于保持机身平稳，减少 roll、pitch 偏差和姿态震荡。核心是 `pose_score_formula`，它直接拟合官方姿态评分逻辑；其他姿态项作为稳定辅助。

```text
B. 姿态稳定与姿态评分拟合
├── pose_score_formula            正向奖励，拟合官方姿态分
├── flat_orientation              负向惩罚，抑制机身倾斜
├── posture_stability             负向惩罚，抑制 roll/pitch 快速变化
├── lin_vel_z                     负向惩罚，抑制机身上下跳动
├── ang_vel_xy                    负向惩罚，抑制 roll/pitch 角速度
└── correct_base_height           负向惩罚，约束机身高度
```

当前关键权重：

```text
pose_score_formula:        3.08
flat_orientation:         -1.65
posture_stability:        -1.65
lin_vel_z:                -0.80
ang_vel_xy:               -0.75
correct_base_height:      -0.45
```

图中建议将 `pose_score_formula` 画成主节点，其他姿态项作为辅助约束节点。其作用是让模型在前进过程中减少跳动、侧倾和俯仰震荡，从而提高姿态得分。

## 4. C. 能耗控制与官方能量评分拟合

该类奖励用于控制关节输出功率和力矩，避免机器人通过高能耗动作换取短期速度。核心是 `energy_score_formula`，它对官方能耗评分进行近似拟合。

```text
C. 能耗控制与能量评分拟合
├── energy_score_formula          正向奖励，拟合官方能耗分
├── energy                        负向惩罚，惩罚功率消耗
├── joint_torques                 负向惩罚，限制大力矩
└── dof_vel                       负向惩罚，限制关节速度过高
```

当前关键权重：

```text
energy_score_formula:      2.42
energy:                   -1.6e-4
joint_torques:            -6.5e-5
dof_vel:                  -2.8e-4
```

图中建议表达为：能耗评分拟合是主目标，功率、力矩和关节速度惩罚是底层约束，用于让机器人学习更省力的运动方式。

## 5. D. 关节动作平滑与机械安全

该类奖励抑制动作突变、关节极限和不自然姿态，提升训练稳定性和真实机器人可执行性。

```text
D. 关节动作平滑与机械安全
├── action_rate                   负向惩罚，抑制相邻动作变化过大
├── action_smoothness             负向惩罚，抑制动作二阶差分过大
├── joint_acc                     负向惩罚，抑制关节加速度
├── dof_pos_limits                负向惩罚，防止关节接近软限位
├── hip_to_default                负向惩罚，约束髋关节偏离默认姿态
└── joint_position_penalty        负向惩罚，约束全关节偏离默认姿态
```

当前关键权重：

```text
action_rate:              -0.023
action_smoothness:        -0.026
joint_acc:                -7.5e-7
dof_pos_limits:           -0.30
hip_to_default:           -0.16
joint_position_penalty:   -0.11
```

图中建议表达为：该模块不是直接追分，而是为策略提供动作可执行性约束，防止模型利用仿真中的极端关节动作。

## 6. E. 足端步态质量与复杂地形通过

该类奖励塑造足端摆动、落脚和接触行为，服务于坡道、楼梯和迷宫入口等复杂地形。

```text
E. 足端步态质量与复杂地形通过
├── feet_air_time                 正向奖励，鼓励适当迈步和离地时间
├── feet_clearance                正向奖励，鼓励摆动脚抬高越过障碍
├── feet_swing_forward            正向奖励，鼓励摆动脚向前迈
├── feet_slide                    负向惩罚，减少脚底打滑
├── feet_stumble                  负向惩罚，减少踢墙、踢台阶边缘
├── air_time_variance_penalty     负向惩罚，鼓励四足步态节奏一致
└── undesired_contacts            负向惩罚，抑制非脚部接触地形
```

当前关键权重：

```text
feet_air_time:             0.34
feet_clearance:            0.16
feet_swing_forward:        0.05
feet_slide:               -0.20
feet_stumble:             -0.07
air_time_variance_penalty:-0.70
undesired_contacts:       -0.30
```

图中建议表达为：该模块负责“怎么迈脚”，与速度模块共同决定机器人是稳定前进、拖脚、跳跃还是卡住。

## 7. F. 目标导航与完成任务

该类奖励是 Track 任务的主任务目标，负责鼓励机器人接近终点并完成赛道。

```text
F. 目标导航与完成任务
├── approach_goal                 正向奖励，每一步接近目标就给奖励
├── goal_distance                 正向奖励，距离目标越近奖励越高
├── goal_backtrack_penalty        负向惩罚，远离目标时扣分
├── task_complete                 大额正向奖励，进入目标完成半径时给奖励
└── directed_exploration          小额正向奖励，鼓励朝目标方向探索新区域
```

当前关键权重：

```text
approach_goal:             18.0
goal_distance:              0.8
goal_backtrack_penalty:    -3.0
task_complete:            220.0
directed_exploration:       0.035
```

图中建议将 `task_complete` 画为最终稀疏目标，将 `approach_goal` 和 `goal_distance` 画为稠密引导信号。二者共同解决长赛道任务中“只靠终点奖励太稀疏”的问题。

## 8. G. 迷宫避障与空间探索

该类奖励只在迷宫或墙体特征明显时发挥主要作用，用于减少撞墙、卡墙和无效绕行。

```text
G. 迷宫避障与空间探索
├── maze_context_gate             诊断门控，用于判断当前是否处于迷宫墙体语境
├── wall_collision                负向惩罚，速度越快撞墙惩罚越重
├── wall_stall_penalty            负向惩罚，贴墙不动或卡墙扣分
├── wall_proximity                负向惩罚，距离墙体过近扣分
├── open_space                    正向奖励，鼓励走向局部开阔区域
└── corridor_centering            负向惩罚，抑制在走廊中严重偏离可通行区域
```

当前关键权重：

```text
maze_context_gate:          0.0
wall_collision:           -12.0
wall_stall_penalty:        -1.8
wall_proximity:            -0.55
open_space:                 0.08
corridor_centering:        -0.30
```

图中建议表达为：该模块是目标导航模块的空间约束层，帮助机器人在迷宫阶段减少撞墙和卡死。

## 9. H. 失败、超时与卡死抑制

该类奖励用于处理训练中的失败状态，防止策略通过倒地、停滞、超时等方式规避其他惩罚。

```text
H. 失败、超时与卡死抑制
├── termination                  负向惩罚，非正常终止扣分
├── navigation_time              每步时间惩罚，鼓励更快完成
├── stuck_penalty                负向惩罚，有速度命令但身体几乎不动
├── stand_still_motion           负向惩罚，零速附近抑制身体和关节晃动
└── commanded_still_penalty      负向惩罚，被命令运动但实际近似不动
```

当前关键权重：

```text
termination:              -5.0
navigation_time:          -0.007
stuck_penalty:            -2.2
stand_still_motion:       -0.85
commanded_still_penalty:  -0.55
```

图中建议表达为：该模块提供底线约束，防止模型通过失败、卡住或慢性超时获得局部最优。

## 10. 奖励体系关系图文本版

```text
Track 总目标：高完成率 + 高时间分 + 高姿态分 + 高能耗分

Track 总奖励
├── 速度主线
│   ├── track_lin_vel_xy
│   ├── command_speed_advantage
│   └── goal_velocity_projection
│
├── 官方评分拟合
│   ├── pose_score_formula
│   └── energy_score_formula
│
├── 稳定性约束
│   ├── flat_orientation
│   ├── posture_stability
│   ├── lin_vel_z
│   ├── ang_vel_xy
│   └── correct_base_height
│
├── 能耗与动作正则
│   ├── energy
│   ├── joint_torques
│   ├── dof_vel
│   ├── joint_acc
│   ├── action_rate
│   └── action_smoothness
│
├── 足端与接触质量
│   ├── feet_air_time
│   ├── feet_clearance
│   ├── feet_swing_forward
│   ├── feet_slide
│   ├── feet_stumble
│   ├── air_time_variance_penalty
│   └── undesired_contacts
│
├── 目标导航
│   ├── approach_goal
│   ├── goal_distance
│   ├── goal_backtrack_penalty
│   ├── task_complete
│   └── directed_exploration
│
├── 迷宫避障
│   ├── maze_context_gate
│   ├── wall_collision
│   ├── wall_stall_penalty
│   ├── wall_proximity
│   ├── open_space
│   └── corridor_centering
│
└── 失败与时间约束
    ├── termination
    ├── navigation_time
    ├── stuck_penalty
    ├── stand_still_motion
    └── commanded_still_penalty
```

## 11. 生图建议

如果用于绘图，建议采用三层拓扑：

```text
第一层：Track 总奖励
第二层：八大模块
第三层：具体奖励函数
```

视觉上可将模块分为四种颜色：

```text
蓝色：速度与目标推进
绿色：姿态、能耗和官方评分拟合
橙色：步态、接触和机械安全
红色：失败、撞墙、卡死和时间惩罚
```

核心主线可以高亮为：

```text
速度命令跟随
→ 目标方向推进
→ 姿态/能耗评分约束
→ 迷宫避障
→ task_complete 完成奖励
```
