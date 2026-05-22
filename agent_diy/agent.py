#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Hybrid Agent: CPG generates base walking + RL fine-tunes + Reflex safety net.

Architecture (aligned with rsl_rl PPO + unitree legged_gym):
  1. CPG controller generates stable trot-gait joint trajectories (PRIMARY)
  2. Navigation module modulates CPG heading/speed (hard-rule)
  3. PPO network outputs residual adjustment (SECONDARY, ±0.3)
  4. Reflex controller overrides on danger (safety only)

Final action = clip(CPG_base + RL_residual, -1, 1) → reflex check
"""

import torch
import numpy as np
import torch.optim as optim

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

from kaiwudrl.interface.agent import BaseAgent
from agent_diy.feature.definition import ActData
from agent_diy.conf.conf import Config
from agent_diy.model.model import ActorCritic
from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.feature.cpg_controller import CPGController
from agent_diy.feature.reflex_controller import ReflexController
from agent_diy.feature.navigation import NavigationController
from tools.train_env_conf_validate import check_usr_conf


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device="cuda", logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        self.logger = logger
        self.monitor = monitor

        usr_conf, usr_conf_file, is_eval, stage = Config.load_conf(self.logger)
        valid, message = check_usr_conf(usr_conf, is_eval, self.logger)
        if not valid:
            raise Exception(f"check_usr_conf failed: {message}")

        self.stage = stage
        self.is_eval = is_eval
        env_conf = usr_conf["env"]
        self.num_envs = env_conf["num_envs"]

        self.num_actions = stage.num_actions
        self.num_critic_obs = stage.num_critic_observations + stage.num_goal_obs
        self.num_obs = stage.num_proprio_obs + stage.num_scan + stage.num_goal_obs

        # ─── Model ───
        self.model = ActorCritic(
            num_obs=self.num_obs,
            num_critic_obs=self.num_critic_obs,
            num_actions=self.num_actions,
            actor_hidden_dims=stage.actor_hidden_dims,
            critic_hidden_dims=stage.critic_hidden_dims,
            activation=stage.activation,
            residual_clip=stage.residual_clip,
        ).to(self.device)

        self.optimizer = optim.Adam([{"params": self.model.parameters(), "name": "actor_critic"}], lr=stage.lr)

        self.algorithm = Algorithm(
            model=self.model, optimizer=self.optimizer, device=self.device,
            logger=self.logger, monitor=self.monitor, learning_rate=stage.lr,
            num_mini_batches=stage.num_mini_batches,
            num_learning_epochs=stage.num_learning_epochs,
        )

        # ─── CPG (primary locomotion) ───
        self.cpg = CPGController(
            num_envs=self.num_envs, device=self.device, dt=0.02,
            base_freq=stage.cpg_base_freq, amp_hip=stage.cpg_amp_hip,
            amp_thigh=stage.cpg_amp_thigh, amp_calf=stage.cpg_amp_calf,
        )

        # ─── Reflex (safety only) ───
        self.reflex = ReflexController(
            num_envs=self.num_envs, device=self.device,
            tip_threshold=stage.reflex_tip_threshold,
            ang_vel_threshold=stage.reflex_ang_vel_threshold,
            recovery_duration=stage.reflex_recovery_duration,
        )

        # ─── Navigation (hard-rule) ───
        self.navigator = NavigationController(num_envs=self.num_envs, device=self.device)

        self.num_steps_per_env = stage.num_steps_per_env
        self.save_interval = stage.model_save_interval

        self.algorithm.init_storage(
            self.num_envs, self.num_steps_per_env,
            actor_obs_shape=(self.num_obs,),
            critic_obs_shape=(self.num_critic_obs,),
            action_shape=(self.num_actions,),
            device=self.device,
        )

        super().__init__(agent_type, device, logger, monitor)

    # ═══════════════════════════════════════════════════════════════
    def predict(self, list_obs_data):
        """
        CPG generates base walking actions, RL outputs residual fine-tuning.
        Returns 7-tuple matching agent_ppo interface.
        """
        (obs, critic_obs) = list_obs_data

        with torch.no_grad():
            projected_gravity = obs[:, 3:6]
            height_scan = obs[:, 45:301]

            goal_obs = None
            if self.stage.num_goal_obs > 0 and obs.shape[1] >= 301 + self.stage.num_goal_obs:
                goal_obs = obs[:, 301:301 + self.stage.num_goal_obs]

            # Nav + slope → CPG modulation
            self.cpg.set_slope_adaptation(projected_gravity)
            nav = self.navigator.compute_navigation(height_scan=height_scan, goal_obs=goal_obs)
            self.cpg.set_modulation(
                turn_bias=nav["turn_bias"] + self.cpg.slope_diag_bias,
                freq_mod=nav["freq_mod"],
            )

            # CPG base actions + advance phase
            cpg_actions = self.cpg.compute_actions(rl_residual=None)

            # RL residual (bounded ±0.3)
            rl_residual, values, log_probs, action_mean, action_std = self.algorithm.act(obs, critic_obs)

            # Combine: CPG base + RL residual
            combined = torch.clamp(cpg_actions + rl_residual, -1.0, 1.0)

            # Reflex safety override
            joint_pos = obs[:, 9:21]
            dones_dummy = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            final_actions, _ = self.reflex.check_and_correct(
                combined, projected_gravity, obs[:, 0:3], joint_pos, dones_dummy,
            )

        # Return 8-tuple: env action + RL residual for PPO consistency
        return (
            final_actions,         # 0: actions for env.step()
            rl_residual,           # 1: RL residual (stored in buffer, NOT final action)
            values,                # 2: value estimates
            log_probs,             # 3: log prob of RL residual
            action_mean,           # 4: policy mean
            action_std,            # 5: policy std
            obs.detach(),          # 6: observation
            critic_obs.detach(),   # 7: critic observation
        )

    def exploit(self, list_obs_data):
        """Deterministic inference for evaluation."""
        (obs) = list_obs_data
        with torch.no_grad():
            projected_gravity = obs[:, 3:6]
            height_scan = obs[:, 45:301]

            goal_obs = None
            if self.stage.num_goal_obs > 0 and obs.shape[1] >= 301 + self.stage.num_goal_obs:
                goal_obs = obs[:, 301:301 + self.stage.num_goal_obs]

            self.cpg.set_slope_adaptation(projected_gravity)
            nav = self.navigator.compute_navigation(height_scan=height_scan, goal_obs=goal_obs)
            self.cpg.set_modulation(
                turn_bias=nav["turn_bias"] + self.cpg.slope_diag_bias,
                freq_mod=nav["freq_mod"],
            )

            _ = self.cpg.compute_actions(rl_residual=None)
            rl_residual = self.model.act_inference(obs)
            combined = torch.clamp(self.cpg.get_prior() + rl_residual, -1.0, 1.0)

            joint_pos = obs[:, 9:21]
            dones_dummy = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            actions, _ = self.reflex.check_and_correct(
                combined, projected_gravity, obs[:, 0:3], joint_pos, dones_dummy,
            )

            return [ActData(action=actions)]

    def learn(self, list_sample_data=None):
        return self.algorithm.learn()

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            return
        pretrained = torch.load(model_file_path, map_location=self.device)
        current_state = self.model.state_dict()
        has_mismatch = any(
            k in current_state and pretrained[k].shape != current_state[k].shape
            for k in pretrained
        )
        if not has_mismatch:
            self.model.load_state_dict(pretrained)
        else:
            self._load_model_partial(self.model, pretrained, model_file_path)
        self.cur_model_name = model_file_path

    def _load_model_partial(self, model, pretrained, path):
        current = model.state_dict()
        loaded, partial, skipped = [], [], []
        for key in current:
            if key not in pretrained:
                skipped.append(key); continue
            old_p, new_p = pretrained[key], current[key]
            if old_p.shape == new_p.shape:
                new_p.copy_(old_p); loaded.append(key)
            else:
                with torch.no_grad():
                    new_p.zero_()
                    sl = tuple(slice(0, min(o, n)) for o, n in zip(old_p.shape, new_p.shape))
                    new_p[sl] = old_p[sl]
                partial.append(f"{key} {list(old_p.shape)}→{list(new_p.shape)}")
        model.load_state_dict(current)
        if self.logger:
            self.logger.info(f"Partial load {path}: {len(loaded)} exact, {len(partial)} partial, {len(skipped)} skipped")
