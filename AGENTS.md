# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Tencent AI Arena KaiwuDRL project for Unitree Go2 quadruped reinforcement learning in Isaac Lab.

- `agent_ppo/`: PPO agent, flat actor-critic model, navigation helpers, rewards, training workflow, and stage TOML configs.
- `agent_diy/`: hybrid CPG + RL residual + reflex agent with its own model, rewards, and workflow.
- `conf/`: global KaiwuDRL application, algorithm, replay buffer, and model dump configuration.
- `isaac_env/`: Isaac Lab environment wrapper and TOML-to-env configuration merge logic.
- `ckpt/`: local checkpoint files and checkpoint metadata.
- `train_test.py`: smoke-test entry point for a single short training run.
- `evaluate_pareto.py`: offline checkpoint ranking from score JSONL/CSV exports.
- Documentation lives in `introduce.md`, `核心训练文档.md`, `奖励函数设定解析.md`, and related Markdown files.

## Build, Test, and Development Commands

Run commands from the repository root. The real simulator stack is Linux-only and expects the external Conda environment `env_isaaclab`.

```bash
python train_test.py
```

Runs a smoke test through the KaiwuDRL framework. Edit `train_test.py` to switch `algorithm_name` between `"ppo"` and `"diy"`.

```bash
python evaluate_pareto.py scores.jsonl --forward-threshold 90 --top-k 10 --plot
python evaluate_pareto.py scores.csv --format csv --pareto-front
```

Analyzes exported evaluation scores and selects Pareto-efficient checkpoints.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and English identifiers. Preserve existing Chinese comments and documentation; do not translate them unless explicitly requested. Keep stage names, TOML filenames, and config class names aligned, for example `TrackNavConfig` with `train_env_conf_track_track_nav.toml`.

Do not move Isaac Lab architecture constants such as `num_actions`, `num_scan`, or `num_critic_observations` into TOML. Reward functions should be named `_reward_<name>()` and configured under `[rewards.<name>]`.

## Testing Guidelines

There is no pytest/unittest suite or coverage requirement. Before platform submission, run `python train_test.py` and verify the code compiles and executes one short training step. For behavioral validation, use platform evaluation, TensorBoard/monitor panels, generated videos with small `num_envs`, and `evaluate_pareto.py`.

## Commit & Pull Request Guidelines

Git history does not define a strict commit format. Use concise, action-oriented commit messages, optionally scoped, such as `ppo: tune track rewards` or `docs: update training notes`.

Pull requests should describe the changed agent or stage, list relevant config files, include smoke-test or evaluation results, and call out any checkpoint compatibility concerns.

## Security & Configuration Tips

Do not commit secrets or platform credentials. Local sync tools such as `ide_sync_server.py` and `codex_rpc_bridge/` are intended for trusted local or IDE-container use only.
