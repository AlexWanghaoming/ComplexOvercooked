import os
import torch as th
import numpy as np
import sys
from typing import List, Dict, Tuple, Optional, Union
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from envs.overcook_gym_env import OvercookPygameEnv
from envs.overcook_mdp import ComplexOvercookedGridworld
from envs.agents import RandomAgent, RLAgent, HumanAgent, LLMAgent
from src.utils.utils import create_parser
import math


def load_agents(p0_type, p1_type, args, rl_checkpoint_path=None, mdp:ComplexOvercookedGridworld=None, env:OvercookPygameEnv=None):

    agent_factories = {
        "random": RandomAgent,
        "rl": lambda: RLAgent(args, rl_checkpoint_path, obs_shape=env.obs_shape),
        "human": lambda: HumanAgent(args),
        "llm": lambda agent_index: LLMAgent(args, mdp, env, agent_index=agent_index),
    }

    p0_agent = agent_factories[p0_type](agent_index=0) if p0_type == "llm" else agent_factories[p0_type]()
    p1_agent = agent_factories[p1_type](agent_index=1) if p1_type == "llm" else agent_factories[p1_type]()
    return p0_agent, p1_agent


def main():
    parser = create_parser()
    parser.add_argument("--map_name", type=str, default="2playerhard")
    # parser.add_argument("--map_name", type=str, default="supereasy")
    parser.add_argument("--rl_checkpoint_path", type=str, default="results/models/vdn_seed7_2playerhard_20250514_203957/best_model")
    # parser.add_argument("--rl_checkpoint_path", type=str, default="results/models/ippo_seed7_supereasy_20250410_100715/best_model")

    parser.add_argument("--p0", type=str, default="llm", choices=["random", "rl", "human", "llm"])
    parser.add_argument("--p1", type=str, default="llm", choices=["random", "rl", "human", "llm"])
    parser.add_argument("--n_episodes", type=int, default=10)

    args = parser.parse_args()
    print("蓝色玩家:", args.p0)
    print("红色玩家:", args.p1)
 

    if args.p0 != "human" and args.p1 != "human":
        env = OvercookPygameEnv(map_name=args.map_name,
                            ifrender=True,
                            debug=False)
    else:
        env = OvercookPygameEnv(map_name=args.map_name,
                            ifrender=True,
                            debug=False,
                            fps=10)
    mdp = ComplexOvercookedGridworld(env)

    if args.p0 == args.p1 == 'human':
        args.both_human = True
    else:
        args.both_human = False

    ep_rewards = []
    for i in range(args.n_episodes):
        nobs, _, available_actions = env.reset()
        p0_agent, p1_agent = load_agents(args.p0, args.p1, args, args.rl_checkpoint_path, mdp=mdp, env=env)
        done = False
        while not done:
            # 更新 HumanAgent 的动作
            if isinstance(p0_agent, HumanAgent):
                p0_agent.update_actions()
            if isinstance(p1_agent, HumanAgent):
                p1_agent.update_actions()

            if isinstance(p0_agent, LLMAgent):
                p0_action = p0_agent.select_action(env, available_actions, agent_idx=0)
            else:
                p0_action = p0_agent.select_action(nobs, available_actions, agent_idx=0)

            if isinstance(p1_agent, LLMAgent):
                p1_action = p1_agent.select_action(env, available_actions, agent_idx=1)
            else:
                p1_action = p1_agent.select_action(nobs, available_actions, agent_idx=1)

            actions = [p0_action, p1_action]
            nobs, _, rewards, dones, infos, available_actions = env.step(actions)
            done = dones[0]
            if done:
                print(infos['episode']['ep_sparse_r'])
                ep_rewards.append(infos['episode']['ep_sparse_r'])
            # print(f"Actions: {actions}")
    print(np.mean(ep_rewards), np.std(ep_rewards)/math.sqrt(args.n_episodes))


if __name__ == "__main__":
    main()