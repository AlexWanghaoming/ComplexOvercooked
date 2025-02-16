import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from src.envs.overcook_pygame.overcook_gym_env import OvercookPygameEnv
import pygame
from pygame import K_w, K_a, K_s, K_d, K_SPACE, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_RSHIFT
import threading
from queue import Queue
import time

class KeyboardThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.actions = np.zeros(2, dtype=np.int32)
        self.prev_keys = pygame.key.get_pressed()
        
        self.player_1 = {K_s: 1, K_d: 2, K_w: 3, K_a: 4, K_SPACE: 5}
        self.player_2 = {K_DOWN: 1, K_RIGHT: 2, K_UP: 3, K_LEFT: 4, K_RSHIFT: 5}
        
    def run(self):
        while self.running:
            curr_keys = pygame.key.get_pressed()
            new_actions = np.zeros(2, dtype=np.int32)

            for key, value in self.player_1.items():
                if curr_keys[key]:
                    new_actions[0] = value
                    break
                    
            for key, value in self.player_2.items():
                if curr_keys[key]:
                    new_actions[1] = value
                    break

            self.actions = new_actions
            self.prev_keys = curr_keys
            
    def get_actions(self):
        return self.actions

def test_keyboard():
    env = OvercookPygameEnv(map_name='supereasy', ifrender=True, debug=True)
    clock = pygame.time.Clock()
    
    keyboard_thread = KeyboardThread()
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    nobs, share_obs, available_actions = env.reset()
    done = False
    step_count = 0
    
    try:
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    return
            
            current_actions = keyboard_thread.get_actions()
            nobs, share_obs, rewards, dones, infos, available_actions = env.step(current_actions)
            done = any(dones)
            step_count += 1
            print(f"Step: {step_count}, Actions: {current_actions}")

            env.render()
            clock.tick(60)

    except (KeyboardInterrupt, SystemExit):
        pass
    
    env.close()
    pygame.quit()

if __name__ == '__main__':
    test_keyboard()