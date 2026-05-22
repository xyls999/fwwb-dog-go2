#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Training workflow — aligned with agent_ppo interface.
CPG generates base walking, RL fine-tunes via PPO, Reflex as safety net.
"""

from common_python.utils.common_func import Frame
import os
import time
from collections import deque, defaultdict
import torch

from agent_diy.conf.conf import Config
from agent_ppo.feature.definition import RolloutStorage
from tools.utils import load_reward_keys_from_monitor_config


def _initialize_training_state(env, agent, logger):
    usr_conf, usr_conf_file, is_eval, stage = Config.load_conf(logger)
    from tools.train_env_conf_validate import check_usr_conf
    valid, message = check_usr_conf(usr_conf, is_eval=False, logger=logger)
    if not valid:
        logger.error(message)
        raise Exception(message)

    agent.algorithm.actor_critic.train()

    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(agent.num_envs, dtype=torch.float, device=agent.device)
    cur_episode_length = torch.zeros(agent.num_envs, dtype=torch.float, device=agent.device)

    storage = agent.algorithm.storage

    data = env.reset(usr_conf)
    if data is None:
        raise Exception("env.reset failed")

    obs, critic_obs = data
    if critic_obs is None:
        critic_obs = obs
    obs = torch.clone(obs)
    critic_obs = torch.clone(critic_obs)
    logger.info(f"obs.shape:{obs.shape}, critic_obs.shape:{critic_obs.shape}")

    reward_keys = load_reward_keys_from_monitor_config()
    logger.info(f"reward_keys: {reward_keys}")

    return (
        storage, obs, critic_obs, ep_infos, rewbuffer, lenbuffer,
        cur_reward_sum, cur_episode_length, reward_keys, usr_conf,
    )


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    agent = agents[0]
    env = envs[0]

    (
        storage, obs, critic_obs, ep_infos, rewbuffer, lenbuffer,
        cur_reward_sum, cur_episode_length, reward_keys, usr_conf,
    ) = _initialize_training_state(env, agent, logger)

    last_obs, last_critic_obs = torch.clone(obs), torch.clone(critic_obs)
    last_report_monitor_time = 0
    episode = 0

    while True:
        logger.info(f"Episode {episode} start")
        start_time = time.time()

        last_obs, last_critic_obs, storage_stats = run_episodes_(
            env, agent, storage, logger,
            last_obs, last_critic_obs, episode,
            ep_infos, cur_reward_sum, cur_episode_length,
            rewbuffer, lenbuffer,
        )
        episode += 1

        agent.learn(list_sample_data=None)
        storage.clear()

        total_cost_time = round(time.time() - start_time, 2)
        logger.info(f"Episode {episode} end, cost_time={total_cost_time}s")

        now = time.time()
        if now - last_report_monitor_time >= 60:
            report_monitor_data(ep_infos, reward_keys, agent, monitor, episode, storage_stats)
            last_report_monitor_time = now
        ep_infos.clear()

        if episode % agent.save_interval == 0:
            agent.save_model()

    env.close()


def run_episodes_(
    env, agent, storage, logger,
    last_obs, last_critic_obs, episode,
    ep_infos, cur_reward_sum, cur_episode_length,
    rewbuffer, lenbuffer,
):
    transition = RolloutStorage.Transition()
    obs, critic_obs = last_obs, last_critic_obs

    with torch.inference_mode():
        for i in range(agent.num_steps_per_env):
            # Predict: CPG base + RL residual + Reflex
            predict_result = agent.predict((obs, critic_obs))
            (
                env_actions, rl_residual, values, actions_log_prob,
                action_mean, action_sigma, detach_obs, detach_critic_obs,
            ) = predict_result

            command_actions = torch.clip(env_actions, -6.0, 6.0).to(agent.device)
            if i == 0:
                logger.info(f"action range: [{command_actions.min():.3f}, {command_actions.max():.3f}]")

            # Step env
            data = env.step(command_actions)
            if data is None:
                raise Exception("env.step failed")

            frame_no, obs, rewards, terminated, truncated, extra = data
            infos, privileged_obs = extra

            critic_obs_new = privileged_obs if privileged_obs is not None else obs
            obs = torch.clone(obs)
            critic_obs_new = torch.clone(critic_obs_new)

            if obs is None:
                raise Exception(f"episode {episode}, obs is None!")

            dones = torch.logical_or(terminated, truncated)

            obs = obs.to(agent.device)
            critic_obs = critic_obs_new.to(agent.device)
            rewards = rewards.to(agent.device)
            dones = dones.to(agent.device)

            # Reset CPG/Nav for done envs
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                agent.cpg.reset_envs(done_ids)
                agent.navigator.reset_envs(done_ids)

            if "episode" in infos:
                ep_infos.append(infos["episode"])

            cur_reward_sum += rewards
            cur_episode_length += 1

            new_ids = (dones > 0).nonzero(as_tuple=False)
            if new_ids.numel() > 0:
                rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0

            if "time_outs" in infos:
                rewards = rewards + agent.algorithm.gamma * torch.squeeze(
                    values * infos["time_outs"].unsqueeze(1).to(agent.device), 1
                )

            # Write transition: store RL residual (what policy sampled), NOT final action.
            # PPO computes log_prob(stored_action); this must match stored actions_log_prob.
            transition.actions = rl_residual
            transition.values = values
            transition.actions_log_prob = actions_log_prob
            transition.action_mean = action_mean
            transition.action_sigma = action_sigma
            transition.observations = detach_obs
            transition.critic_observations = detach_critic_obs
            transition.rewards = rewards.clone()
            transition.dones = dones

            storage.add_transitions(transition)
            transition.clear()

        # GAE
        with torch.no_grad():
            last_values = agent.algorithm.actor_critic.evaluate(
                torch.clone(critic_obs).detach()
            ).detach()
        storage.compute_returns(last_values, agent.algorithm.gamma, agent.algorithm.lam)

        storage_stats = {
            "reward_mean": storage.rewards.mean().item(),
            "reward_std": storage.rewards.std().item(),
        }

        last_obs = torch.clone(obs)
        last_critic_obs = torch.clone(critic_obs)

    return last_obs, last_critic_obs, storage_stats


def report_monitor_data(ep_infos, reward_keys, agent, monitor, episode, storage_stats=None):
    monitor_data = {"episode_cnt": episode}
    if storage_stats:
        monitor_data["reward_mean"] = storage_stats.get("reward_mean", 0.0)
        monitor_data["reward_std"] = storage_stats.get("reward_std", 0.0)
    if ep_infos:
        gm = defaultdict(list)
        for ep in ep_infos:
            for key in reward_keys:
                if key in ep:
                    m = ep[key]
                    if not isinstance(m, torch.Tensor):
                        m = torch.tensor(m, device=agent.device)
                    gm[key].append(m.float().mean())
        for k, v in gm.items():
            if v:
                monitor_data[k] = torch.stack(v).mean().item()
        monitor_data["episode_reward"] = sum(monitor_data.get(k, 0) for k in reward_keys)
    monitor.put_data({os.getpid(): monitor_data})
