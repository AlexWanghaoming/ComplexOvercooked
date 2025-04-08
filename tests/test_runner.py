import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../src/")
from src.runners.episode_runner import linear_annealed_factor  

if __name__ == "__main__":
    print("\n| test_runner.py")
    max_t = 100
    for t in range(1, max_t):
        f = linear_annealed_factor(t_env=t, start_value=1, end_value=0, anneal_start=1, anneal_end=max_t)
        print(f)