# Go2 RL Gym — Agent Guide

> This file is written for AI coding agents. The project combines English code/naming with Chinese comments and documentation. When modifying code, preserve existing comment languages.

## Project Overview

**Go2 RL Gym** is a reinforcement learning training and deployment framework for the Unitree Go2 quadruped robot. It is built on top of [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) and [legged_gym](https://github.com/leggedrobotics/legged_gym), with custom extensions for Concurrent Teacher-Student (CTS) training and Mixture-of-Experts (MoE) policies.

The repository supports:
- **Training** in NVIDIA Isaac Gym (GPU-parallelized simulation)
- **Sim2Sim** validation in MuJoCo
- **Sim2Real** deployment on the physical Go2 robot (Python or C++)
- **Asynchronous evaluation** via the RoboGauge framework during training

Related paper: [Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion](https://arxiv.org/abs/2602.00678) (arXiv:2602.00678).

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Deep Learning | PyTorch 2.3.1, torchvision 0.18.1 |
| Simulation (Training) | NVIDIA Isaac Gym (Preview 4) |
| Simulation (Validation) | MuJoCo 3.2.3 |
| RL Algorithms | PPO, CTS, MoE-CTS, MCP-CTS, AC-MoE-CTS, Dual-MoE-CTS |
| Physics Engine | PhysX (via Isaac Gym) |
| Logging | TensorBoard 2.14.0 |
| Export formats | TorchScript (`.pt`), ONNX (`.onnx`), pickle (`.pkl`) |
| Real robot SDK | `unitree_sdk2_python` (Python), `unitree_sdk2` + `unitree_cpp_deploy` (C++) |

### Key Dependencies

- `isaacgym` — must be installed manually from NVIDIA's official package
- `rsl-rl` — bundled in `rsl_rl/`, install with `pip install -e rsl_rl/`
- `numpy==1.20`, `matplotlib`, `pyyaml`, `onnx==1.17.0`, `pygame`, `mujoco==3.2.3`
- Optional: `robogauge` for async Sim2Sim evaluation

## Project Structure

```
go2_rl_gym/
├── setup.py                          # Root package setup (go2_rl_gym)
├── legged_gym/
│   ├── __init__.py                   # Defines LEGGED_GYM_ROOT_DIR
│   ├── envs/
│   │   ├── base/
│   │   │   ├── base_task.py          # Isaac Gym task base class
│   │   │   ├── legged_robot.py       # Core robot environment (step, reset, rewards, terrain)
│   │   │   └── legged_robot_config.py# Base configs: PPO, CTS, MoE-CTS variants
│   │   └── go2/
│   │       ├── go2_env.py            # Go2-specific observation & reward overrides
│   │       ├── go2_config.py         # Go2 env + training configs for all algorithms
│   │       └── go2_config_*.py       # Ablation / variant configs
│   ├── scripts/
│   │   ├── train.py                  # Training entrypoint
│   │   └── play.py                   # Visualization + policy export entrypoint
│   └── utils/
│       ├── task_registry.py          # Registers (task_name -> env_class, env_cfg, train_cfg)
│       ├── helpers.py                # Args parsing, seeding, cfg override, load path helpers
│       ├── exporter.py               # TorchScript / ONNX / PKL policy export logic
│       ├── terrain.py                # Procedural terrain generation (trimesh/heightfield)
│       └── logger.py                 # Training logger
├── rsl_rl/                           # RL algorithm library (separate pip package)
│   ├── setup.py
│   └── rsl_rl/
│       ├── algorithms/               # PPO, CTS, MoECTS, MoENGCTS, MCPCTS, ACMoECTS, DualMoECTS
│       ├── modules/                  # Actor-Critic networks for each algorithm variant
│       ├── runners/                  # OnPolicyRunner, OnPolicyRunnerCTS
│       ├── storage/                  # RolloutStorage, RolloutStorageCTS
│       └── env/vec_env.py            # VecEnv protocol
├── deploy/
│   ├── pre_train/go2/                # Pre-trained checkpoints (.pt)
│   ├── deploy_mujoco/                # Sim2Sim MuJoCo deployment
│   │   ├── deploy_go2.py             # Loads TorchScript policy, runs in MuJoCo with gamepad support
│   │   ├── utils.py                  # Rendering / video recording helpers
│   │   └── configs/go2.yaml          # MuJoCo deployment config (policy path, PD gains, scales)
│   └── deploy_real/                  # Sim2Real Python deployment
│       ├── deploy_real_go2.py        # Runs on Jetson / onboard PC via unitree_sdk2_python
│       ├── config_go2.py             # Real robot config parser
│       ├── configs/go2.yaml          # Real deployment parameters
│       └── common/                   # Command helpers, rotation utils, remote controller parser
├── resources/robots/go2/             # URDF, MJCF, terrain XMLs for MuJoCo
├── tools/
│   ├── logs_compress.py              # Compress logs (excluding .pt) with zstd tar
│   └── logs_merge.py                 # Merge RoboGauge YAML results into CSV
└── doc/
    ├── setup_en.md                   # English installation guide
    └── setup_zh.md                   # Chinese installation guide
```

## Installation & Build

This project requires **Ubuntu 18.04+** with an **NVIDIA GPU** (driver 525+). It is **not** supported on Windows or macOS.

### 1. Environment Setup

```bash
conda create -n unitree-rl python=3.8
conda activate unitree-rl
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2. Install Isaac Gym

Download from [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym), then:

```bash
cd isaacgym/python
pip install -e .
# Verify: cd examples && python 1080_balls_of_solitude.py
```

### 3. Install Project Packages

```bash
# Install rsl_rl
cd rsl_rl
pip install -e .

# Install go2_rl_gym
cd ..
pip install -e .
```

### 4. Optional Components

- **RoboGauge evaluation**: `pip install -e .` in [RoboGauge](https://github.com/wty-yy/RoboGauge) clone
- **Real robot Python deploy**: `pip install -e .` in `unitree_sdk2_python` clone
- **Real robot C++ deploy**: follow [unitree_cpp_deploy](https://github.com/wty-yy/unitree_cpp_deploy)

## Running the Code

### Training

```bash
# Standard PPO
python legged_gym/scripts/train.py --task=go2 --num_envs 4096 --headless

# MoE-CTS (paper final version)
python legged_gym/scripts/train.py --task=go2_moe_cts --num_envs 8192 --headless

# Resume from checkpoint
python legged_gym/scripts/train.py --task=go2_moe_cts --num_envs 8192 --headless --resume --load_run Mar21_22-54-5-46_

# With RoboGauge async evaluation
python legged_gym/scripts/train.py --task=go2_moe_cts --num_envs 8192 --headless --robogauge --robogauge_port 9973
```

Available tasks: `go2`, `go2_cts`, `go2_moe_cts`, `go2_moe_ng_cts`, `go2_mcp_cts`, `go2_ac_moe_cts`, `go2_dual_moe_cts`.

### Play / Visualization & Export

```bash
python legged_gym/scripts/play.py --task=go2_moe_cts --num_envs 100
```

Play mode:
- Loads the latest checkpoint automatically (override with `--load_run`, `--checkpoint`)
- Disables domain randomization and noise for deterministic evaluation
- Exports the actor policy to:
  - `logs/<experiment_name>/exported/policies/policy.pt` (TorchScript)
  - `logs/<experiment_name>/exported/policies/policy.onnx` (ONNX)
  - `logs/<experiment_name>/exported/policies/policy.pkl` (raw weights)

### Sim2Sim (MuJoCo)

```bash
python deploy/deploy_mujoco/deploy_go2.py
```

- Supports Xbox-compatible gamepad teleoperation (auto-detected via pygame)
- Edit `deploy/deploy_mujoco/configs/go2.yaml` to change `policy_path` or terrain XML
- Available terrains: `flat.xml`, `stairs.xml`, `race_track.xml`, `cross_stairs.xml`, `cross_slope.xml`

### Sim2Real (Python)

On the robot's onboard computer (Jetson, JetPack 5/6):

```bash
cd deploy/deploy_real
python deploy_real_go2.py eth0
```

- Requires `unitree_sdk2_python` installed
- In Unitree app: disable `mcf/*`, enable `ota_box` service
- Press `start` to stand, `A` to engage controller, `select` to exit

## Code Organization & Conventions

### Task Registration

All environments are registered in `legged_gym/envs/__init__.py` via `task_registry.register(name, env_class, env_cfg, train_cfg)`. The registry is consumed by `train.py` and `play.py`.

### Configuration System

Configs use nested Python classes inheriting from `BaseConfig` (in `base_config.py`). Example hierarchy:

- `LeggedRobotCfg` — base environment config (terrain, rewards, domain rand, control, asset)
- `LeggedRobotCfgPPO` — base PPO training config
- `LeggedRobotCfgCTS` — base CTS training config
- `GO2Cfg` — Go2-specific env overrides
- `GO2CfgMoECTS` — Go2 + MoE-CTS training overrides

Configs are converted to dicts via `class_to_dict()` for YAML serialization and runner consumption.

### Adding a New Algorithm Variant

1. **Network**: Add `ActorCritic<Name>` in `rsl_rl/rsl_rl/modules/`
2. **Algorithm**: Add `<Name>` in `rsl_rl/rsl_rl/algorithms/` (inherit from `CTS` or `PPO`)
3. **Runner**: If storage/computation differs, extend `OnPolicyRunnerCTS` or add logic in existing runner
4. **Config**: Add `LeggedRobotCfg<Name>` in `legged_gym/envs/base/legged_robot_config.py`
5. **Go2 Config**: Add `GO2Cfg<Name>` in `legged_gym/envs/go2/go2_config.py`
6. **Register**: Add `task_registry.register("go2_<name>", Go2Robot, GO2Cfg(), GO2Cfg<Name>())` in `legged_gym/envs/__init__.py`
7. **Export**: Update `legged_gym/utils/exporter.py` `_TorchPolicyExporter` and `_OnnxPolicyExporter` if the forward signature differs

### Observation Structure (Go2)

Student observation (`num_observations = 45`):
- `[0:3]` angular velocity (scaled)
- `[3:6]` projected gravity
- `[6:9]` commands (lin_vel_x, lin_vel_y, ang_vel_yaw)
- `[9:21]` DOF position errors (scaled)
- `[21:33]` DOF velocities (scaled)
- `[33:45]` previous actions

Privileged observation (`num_privileged_obs = 263`):
- Student obs (45)
- Base linear velocity (3)
- Foot contact forces (4)
- Motor torques / limits (12)
- Motor accelerations (12)
- Height measurements (187)

### Reward Functions

Rewards are auto-discovered via `self._reward_<name>()` methods in `LeggedRobot`. Scales are defined in `cfg.rewards.scales`. Non-zero scales are multiplied by `dt` during initialization. Curriculum scaling can be applied via `cfg.rewards.curriculum_rewards`.

### Domain Randomization

Randomizations are applied at different frequencies:
- **Per environment creation**: friction, restitution
- **Per reset (`reset_idx`)**: motor strength, motor zero offset, PD gains, base mass, link mass, base COM
- **Per step**: action delay (probabilistic), push robots (interval-based)

## Testing & Quality Assurance

This is a **research codebase** without unit tests or CI pipelines. Validation is done through:

1. **Sim2Sim in MuJoCo** — primary validation before real deployment
2. **RoboGauge async evaluation** — automated Sim2Sim benchmark during training (tracking, safety, quality, terrain robustness)
3. **Play mode visualization** — human inspection in Isaac Gym
4. **Real robot trials** — final validation on physical hardware

When making changes:
- Run `play.py` with a small number of envs (`--num_envs 8`) to verify basic functionality
- Test in MuJoCo with `deploy_go2.py` before claiming sim2real readiness
- Check TensorBoard logs for reward trends, terrain level progression, and value loss stability

## Deployment Artifacts

### Checkpoint Format

Saved by `OnPolicyRunnerCTS.save()`:
```python
{
    'model_state_dict': ...,
    'optimizer1_state_dict': ...,
    'optimizer2_state_dict': ...,
    'iter': ...,
    'infos': ...,
}
```

### Export Forward Signatures

- **CTS**: `action, (None, latent)`
- **MoE-CTS / AC-MoE-CTS / Dual-MoE-CTS**: `action, (weights, latent)`
- **MoE-NG-CTS / MCP-CTS**: `action, (weights, latent)` with `obs_no_goal_mask` handling

The exporter automatically detects the policy type via `hasattr` checks on `student_encoder`, `student_moe_encoder`, `actor_mcp`, etc.

## Security Considerations

- The real robot deployment code (`deploy_real_go2.py`) sends low-level motor commands over DDS. Always test in simulation first.
- `unitree_sdk2_python` requires network interface access (e.g., `eth0`). Ensure you are on the correct robot network.
- No secrets or credentials are stored in this repository.

## Common Pitfalls for Agents

1. **Do not assume Windows compatibility.** Isaac Gym is Linux-only.
2. **Always check `task_registry` registration** when adding new tasks — missing registration causes `ValueError` at runtime.
3. **Preserve comment languages.** Many comments are in Chinese; do not translate them unless explicitly asked.
4. **NumPy version is pinned to 1.20.** Newer NumPy may cause compatibility issues with Isaac Gym.
5. **Isaac Gym is deprecated.** Do not attempt to upgrade to Isaac Sim without explicit instruction — the APIs differ significantly.
6. **MoE student encoder has two optimizers.** CTS-based runners use `optimizer1` (teacher + actor + critic) and `optimizer2` (student encoder). Save/load must handle both.
7. **ONNX export flattens history by observation terms**, not by time steps. The C++ deployment stack expects this specific flattening order.
