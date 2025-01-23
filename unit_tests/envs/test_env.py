import numpy as np
# export PYTHONPATH="$PWD"
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from src.envs.overcook_pygame.overcook_gym_env import OvercookPygameEnv

def test_overcooked2_env():
    print("\n| test_overcooked2_env()")
    # env = OvercookPygameEnv(map_name='4playersplit', ifrender=True, debug=True)
    # env = OvercookPygameEnv(map_name='2playerhard', ifrender=True, debug=True)
    env = OvercookPygameEnv(map_name='supereasy', ifrender=True, debug=True)

    # print(env.get_state()[0].shape[0])
    # print(env.observation_space[0].shape)
    # print(env.action_space)
    # print(env.observation_space)
    # print(env.share_observation_space)
    # print(env.episode_limit)
    # assert isinstance(env.env_name, str)
    # assert isinstance(env.state_dim, int)
    # assert isinstance(env.action_dim, int)
    nobs, share_obs, available_actions = env.reset()
    assert isinstance(nobs, list)
    assert isinstance(share_obs, np.ndarray)
    print("observation shape:", nobs[0].shape)
    done = False
    while not done:
        random_action = np.random.randint(0, 6, size=env.n_agents)
        nobs, share_obs, rewards, dones, infos, available_actions = env.step(random_action)
        assert isinstance(nobs, list)
        assert isinstance(share_obs, np.ndarray)
        assert isinstance(rewards, list)
        assert isinstance(dones, list)
        assert isinstance(infos, dict)
        assert isinstance(available_actions, list)

        done = dones[0]

if __name__ == '__main__':
    print('\n| test_env.py')
    test_overcooked2_env()