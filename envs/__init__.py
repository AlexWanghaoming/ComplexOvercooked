from gym.envs.registration import register
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')


register(
    id='Overcooked_pygame-zhenghe-v2',
    # entry_point="overcook_gym:make_env",
    entry_point="overcook_pygame.overcook_gym_env:OvercookPygameEnv",
)