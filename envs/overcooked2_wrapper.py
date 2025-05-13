from overcook_gym_env import OvercookPygameEnv
from .multiagentenv import MultiAgentEnv
from gymnasium.spaces import flatdim
import numpy as np
import torch as th
from collections.abc import Iterable
import warnings

class Overcooked2Wrapper(MultiAgentEnv):
    def __init__(self, map_name, seed, common_reward=True, reward_scalarisation='mean', **kwargs):
        self.env = OvercookPygameEnv(map_name=map_name, 
                                     seed=seed, 
                                     **kwargs)
        self.episode_limit = self.env.episode_limit
        self.n_agents = self.env.n_agents
        # print(env.action_space)
        # print(env.observation_space)
        # print(env.share_observations_space)
        self.obs = None
        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation} (only support 'sum' or 'mean')"
                )
        
    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        if isinstance(actions, th.Tensor):
            actions = actions.cpu().numpy()
        elif isinstance(actions, np.ndarray):
            actions = actions
        else:
            raise ValueError(f"actions should be np.ndarray or th.Tensor, got {type(actions)}")
        if self.env.debug:
            print("actions:", actions)

        self.nobs, share_obs, rewards, dones, infos, available_actions = self.env.step(actions)

        truncated = False
        # print(rewards)
        if self.common_reward and isinstance(rewards, Iterable):
            reward = float(self.reward_agg_fn(rewards))
        elif not self.common_reward and not isinstance(rewards, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment, returning reward as is"
            )

        if isinstance(dones, Iterable):
            done = all(dones)

        return self.nobs, reward, done, truncated, infos

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.nobs
    
    def get_obs_agent(self, agent_id:int):
        """Returns observation for agent_id"""
        return self.env.get_obs()[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        print("shape of observation:", self.env.obs_shape)
        return self.env.obs_shape

    def get_state(self):
        # return np.concatenate(self.env.get_obs(), axis=0).astype(np.float32)
        return self.env.get_obs()[0]

    def get_state_size(self):
        """Returns the shape of the state"""
        # return self.env.obs_shape * self.n_agents
        state_shape = self.get_state().shape[0]
        print("shape of state:", state_shape)
        return state_shape

    def get_avail_actions(self):
        return self.env.get_avail_actions()
        # return [[1]*self.get_total_actions()]*self.n_agents
    
    def get_avail_agent_actions(self, agent_id:int):
        """Returns the available actions for agent_id"""
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.env.action_space[0].n

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        if seed is not None:
            self.env.seed(seed)
        self.nobs, share_obs, available_actions = self.env.reset()
        return self.nobs, share_obs, available_actions

    def render(self):
        raise NotImplementedError

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_replay(self):
        self.env.save_replay()

    def get_env_info(self):
        env_info = super().get_env_info()
        return env_info

    def get_stats(self):
        return {}
