import os
import torch as th
import numpy as np
import sys
from typing import List, Dict, Tuple, Optional, Union
from overcook_gym_env import OvercookPygameEnv
from llm_agent import LlmMediumLevelAgent

from src.modules.agents.rnn_agent import RNNAgent
from src.components.action_selectors import SoftPoliciesSelector, EpsilonGreedyActionSelector
import argparse
import random
import pygame
from pygame import K_w, K_a, K_s, K_d, K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_m


class RLAgent:
    def __init__(self,args, checkpoint_path, obs_shape):
        self.args = args    
        if self.args.obs_agent_id:
            input_shape = obs_shape + args.n_agents
        self.agent = RNNAgent(input_shape = input_shape, args = args)
        self.load_model(checkpoint_path)
        self.args = args    
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(1, self.args.n_agents, -1)
        self.agent_output_type = "pi_logits"
        self.action_selector = SoftPoliciesSelector(args) 
        self.deterministic = False

    def load_model(self, checkpoint_path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(checkpoint_path), map_location="cpu"))

    def select_action(self, obs:np.ndarray, avail_actions:List, agent_idx:int):
        if self.args.obs_agent_id:
            inputs = np.hstack((obs, np.eye(self.args.n_agents)))

        inputs = th.tensor(inputs, dtype=th.float32)
        with th.no_grad():
            agent_outs, self.hidden_states = self.agent(inputs, hidden_state=self.hidden_states)
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            # Make the logits for unavailable actions very negative to minimise their affect on the softmax
            reshaped_avail_actions = np.array(avail_actions).reshape(self.args.n_agents, -1)
            agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1).view(1, self.args.n_agents, -1)
            if self.deterministic:
                chosen_actions = th.argmax(agent_outs[0][0]).item()
            else:
                chosen_actions = self.action_selector.select_action(agent_inputs=agent_outs)
                chosen_actions = chosen_actions[0][agent_idx].item()

            return chosen_actions
        

class HumanAgent:
    def __init__(self, args):
        self.both_human = args.both_human
        self.current_actions = [0, 0]  # 用于存储两个玩家的动作
        self.keyboard_mapping = {
            'a': {K_s: 1, K_d: 2, K_w: 3, K_a: 4, K_SPACE: 5},
            'b': {K_DOWN: 1, K_RIGHT: 2, K_UP: 3, K_LEFT: 4, K_m: 5}
        }
        # 设置键盘检测的时间间隔
        self.key_check_interval = 10  # 单位：毫秒

        # 初始化上一次键盘检测的时间
        self.last_key_check_time = pygame.time.get_ticks()

    def update_actions(self):
        current = pygame.time.get_ticks()
        if current - self.last_key_check_time >= self.key_check_interval:
            self.last_key_check_time = current
            keys_pressed = pygame.key.get_pressed()

            # 一次性检测所有按键
            p1_keys = [key for key, action in self.keyboard_mapping['a'].items() if keys_pressed[key]]
            p2_keys = [key for key, action in self.keyboard_mapping['b'].items() if keys_pressed[key]]

            # 如果有按键被按下，获取对应的动作
            if self.both_human:
                self.current_actions[0] = self.keyboard_mapping['a'][p1_keys[-1]] if p1_keys else 0
                self.current_actions[1] = self.keyboard_mapping['b'][p2_keys[-1]] if p2_keys else 0
            else:
                self.current_actions[0] = self.current_actions[1] = self.keyboard_mapping['b'][p2_keys[-1]] if p2_keys else 0
        else:
            self.current_actions[0] = self.current_actions[1] = 0
            
    def select_action(self, obs: np.ndarray, avail_actions: list, agent_idx: int):
        return self.current_actions[agent_idx]
    

class LLMAgent:
    def __init__(self, args, mdp, env, agent_index):
        self.agent = LlmMediumLevelAgent(mdp=mdp, env=env, agent_index=agent_index, layout=args.map_name)

    def select_action(self, env, avail_actions:List, agent_idx:int):
        return self.agent.action(env=env)

    
    
class RandomAgent:
    def select_action(self, obs:np.ndarray, avail_actions:List, agent_idx:int):
        consider_actions = [i for i, x in enumerate(avail_actions[agent_idx]) if x == 1]
        return random.choice(consider_actions)
    