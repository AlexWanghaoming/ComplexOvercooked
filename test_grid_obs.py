#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from envs.overcook_gym_env import OvercookPygameEnv

def test_grid_obs():
    """测试网格观测编码方法"""
    
    # 创建环境
    env = OvercookPygameEnv(map_name="supereasy", ifrender=False, debug=True)
    
    # 重置环境
    obs, share_obs, available_actions = env.reset()
    
    print("=== 测试网格观测编码 ===")
    print(f"地图名称: {env.map_name}")
    print(f"玩家数量: {env.n_agents}")
    print(f"物品数量: {len(env.itemdict)}")
    print(f"任务数量: {len(env.taskdict)}")
    
    # 获取网格观测
    grid_obs = env.get_obs_grid()
    
    print(f"\n网格观测形状:")
    for i, obs in enumerate(grid_obs):
        print(f"玩家 {i}: {obs.shape}")
    
    # 计算通道数
    layout = env.game.layout if hasattr(env.game, 'layout') else maps[env.map_name]['layout']
    h, w = len(layout), max(len(row) for row in layout)
    expected_channels = (1 + env.n_agents + 1 + 1 + 1 + len(env.itemdict) + 3 + 3 + len(env.taskdict))
    
    print(f"\n地图尺寸: {h} x {w}")
    print(f"预期通道数: {expected_channels}")
    print(f"实际通道数: {grid_obs[0].shape[2]}")
    
    # 显示地图布局
    print(f"\n地图布局:")
    for i, row in enumerate(layout):
        print(f"行 {i}: {row}")
    
    # 显示每个通道的信息
    print(f"\n=== 通道信息 ===")
    channel_names = [
        "墙壁",
        *[f"玩家{i}" for i in range(env.n_agents)],
        "锅位置",
        "案板位置", 
        "收银台位置",
        *[f"物品:{item}" for item in env.itemdict.keys()],
        "锅状态-空", "锅状态-烹饪中", "锅状态-完成",
        "案板状态-空", "案板状态-切菜中", "案板状态-完成",
        *[f"任务:{task}" for task in env.taskdict.keys()]
    ]
    
    for i, name in enumerate(channel_names):
        print(f"通道 {i}: {name}")
    
    # 显示玩家0的观测中非零元素
    print(f"\n=== 玩家0观测中的非零元素 ===")
    obs_0 = grid_obs[0]
    non_zero_positions = np.where(obs_0 > 0)
    
    for y, x, c in zip(non_zero_positions[0], non_zero_positions[1], non_zero_positions[2]):
        value = obs_0[y, x, c]
        print(f"位置({y}, {x}) 通道{c}({channel_names[c]}): {value}")
    
    # 测试几步游戏
    print(f"\n=== 测试游戏步骤 ===")
    for step in range(3):
        # 随机动作
        actions = [np.random.randint(0, 6) for _ in range(env.n_agents)]
        
        # 执行动作
        obs, share_obs, rewards, dones, infos, available_actions = env.step(actions)
        
        # 获取新的网格观测
        grid_obs_new = env.get_obs_grid()
        
        print(f"步骤 {step + 1}:")
        print(f"  动作: {actions}")
        print(f"  奖励: {rewards}")
        print(f"  网格观测形状: {[obs.shape for obs in grid_obs_new]}")
        
        # 检查玩家位置是否在网格中正确编码
        for player_id, player in enumerate(env.game.playergroup):
            player_x, player_y = player.rect.x // 80, player.rect.y // 80
            if 0 <= player_y < h and 0 <= player_x < w:
                player_channel = 1 + player_id  # 玩家通道从1开始
                if grid_obs_new[0][player_y, player_x, player_channel] == 1.0:
                    print(f"  玩家{player_id}位置({player_x}, {player_y})正确编码")
                else:
                    print(f"  玩家{player_id}位置({player_x}, {player_y})编码错误")
    
    print(f"\n测试完成!")

if __name__ == "__main__":
    test_grid_obs() 