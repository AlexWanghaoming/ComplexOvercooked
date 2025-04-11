import numpy as np
import sys,os
import openai
from typing import List, Tuple
import json
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from envs.overcook_gym_env import OvercookPygameEnv
from envs.llm_agent import LlmMediumLevelAgent 
from envs.overcook_mdp import ComplexOvercookedGridworld
import pygame

class LLMController:
    def __init__(self):
        # 设置OpenAI API密钥
        openai.api_key = "your-api-key-here"
        self.actions = np.zeros(2, dtype=np.int32)
        
        # 动作映射
        self.action_mapping = {
            "stay": 0,
            "down": 1,
            "right": 2,
            "up": 3,
            "left": 4,
            "interact": 5
        }
        
        # 系统提示词
        self.system_prompt = """你是一个游戏AI助手，需要控制两个角色在过度烹饪(Overcooked)游戏中协作。
每个角色可以执行以下动作之一：
- stay: 原地不动
- down: 向下移动
- right: 向右移动
- up: 向上移动
- left: 向左移动
- interact: 与面前的物品/设备交互（拾取、放下、切菜、烹饪等）

请严格按照以下JSON格式输出两个角色的动作：
{
    "player1": "<动作>",
    "player2": "<动作>"
}
"""

    def get_actions(self, game_state: dict) -> np.ndarray:
        # 构建用户提示词
        user_prompt = f"""
当前游戏状态：
玩家1：
- 位置：{game_state['player1_pos']}
- 朝向：{game_state['player1_direction']}
- 手持物品：{game_state['player1_item']}
- 是否持有盘子：{game_state['player1_has_dish']}
- 是否在切菜：{game_state['player1_is_cutting']}

玩家2：
- 位置：{game_state['player2_pos']}
- 朝向：{game_state['player2_direction']}
- 手持物品：{game_state['player2_item']}
- 是否持有盘子：{game_state['player2_has_dish']}
- 是否在切菜：{game_state['player2_is_cutting']}

当前订单：{game_state['orders']}
当前得分：{game_state['score']}
剩余时间：{game_state['time_left']}

请为两个玩家选择合适的动作，考虑以下因素：
1. 如果玩家正在切菜，只能选择stay或interact
2. 移动时需要考虑与其他玩家的碰撞
3. 优先完成时间最短的订单
4. 尽量让两个玩家协作，避免同时去做同一件事
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            # 解析LLM响应
            action_text = response.choices[0].message.content
            action_dict = json.loads(action_text)
            
            # 将文本动作转换为数值动作
            new_actions = np.zeros(2, dtype=np.int32)
            new_actions[0] = self.action_mapping[action_dict["player1"]]
            new_actions[1] = self.action_mapping[action_dict["player2"]]
            
            return new_actions
            
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return np.zeros(2, dtype=np.int32)

def test_llm_control():
    env = OvercookPygameEnv(map_name='supereasy', ifrender=True, debug=True)
    clock = pygame.time.Clock()
    
    
    nobs, share_obs, available_actions = env.reset()
    mdp = ComplexOvercookedGridworld(env)
    llm_controller = LlmMediumLevelAgent(mdp=mdp, env=env, agent_index=0)
    llm_controller.action(state=env.state)
    done = False
    step_count = 0
    
    try:
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    return
            

            # 获取LLM决策的动作
            # current_actions = llm_controller.
            
            # 执行动作
            # nobs, share_obs, rewards, dones, infos, available_actions = env.step(current_actions)
            # done = any(dones)
            # step_count += 1
            # print(f"Step: {step_count}, Actions: {current_actions}")

            env.render()
            clock.tick(1)  # 降低帧率以适应API调用延迟

    except (KeyboardInterrupt, SystemExit):
        pass
    
    env.close()
    pygame.quit()

if __name__ == '__main__':
    test_llm_control()