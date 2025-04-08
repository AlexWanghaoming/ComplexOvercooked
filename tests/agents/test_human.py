import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from src.envs.overcook_pygame.overcook_gym_env import OvercookPygameEnv
import pygame
from pygame import K_w, K_a, K_s, K_d, K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_m
import threading
from queue import Queue
import time


def test_keyboard():
    env = OvercookPygameEnv(map_name='supereasy', ifrender=True, debug=True)
    nobs, share_obs, available_actions = env.reset()
    clock = pygame.time.Clock()

    # 设置键盘检测的时间间隔
    key_check_interval = 100  # 单位：毫秒

    # 初始化上一次键盘检测的时间
    last_key_check_time = pygame.time.get_ticks()

    game_over = False


    keyboard_mapping = {
        'a': {K_s: 1, K_d: 2, K_w: 3, K_a: 4, K_SPACE: 5},
        'b': {K_DOWN: 1, K_RIGHT: 2, K_UP: 3, K_LEFT: 4, K_m: 5}
    }
    reward = 0
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        current = pygame.time.get_ticks()

        if current - last_key_check_time >= key_check_interval:
            last_key_check_time = current
            keys_pressed = pygame.key.get_pressed()
            
            # 一次性检测所有按键
            p1_keys = [key for key, action in keyboard_mapping['a'].items() if keys_pressed[key]]
            p2_keys = [key for key, action in keyboard_mapping['b'].items() if keys_pressed[key]]
            
            # 如果有按键被按下，获取对应的动作
            p1_action = keyboard_mapping['a'][p1_keys[-1]] if p1_keys else 0
            p2_action = keyboard_mapping['b'][p2_keys[-1]] if p2_keys else 0
            
            nobs, share_obs, rewards, dones, infos, available_actions = env.step((p1_action, p2_action))
        else:
            nobs, share_obs, rewards, dones, infos, available_actions = env.step((0, 0))
        print(rewards)
        reward+=rewards[0]
        env.game.update_reward(reward)
        if dones[0]:
            game_over = True

        pygame.display.update()
        pygame.display.flip()

        # 控制游戏帧率
        clock.tick(10) # 1s 10 timesteps

    # 退出pygame
    pygame.quit()


if __name__ == '__main__':
    test_keyboard()