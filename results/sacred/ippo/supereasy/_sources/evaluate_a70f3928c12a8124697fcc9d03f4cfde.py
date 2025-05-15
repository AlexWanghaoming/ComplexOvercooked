from copy import deepcopy
import os
from os.path import dirname, abspath
import sys
import yaml
import numpy as np
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import torch as th

from utils.logging import get_logger
from run import run
from utils.utils import config_copy, get_config, recursive_dict_update


SETTINGS["CAPTURE_MODE"] = (
    "fd"  # set to "no" if you want to see stdout/stderr in console
)

logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config["env_args"]["seed"] = config["seed"]

    # run the framework
    run(_run, config, _log)


if __name__ == "__main__":
    params = deepcopy(sys.argv)
    th.set_num_threads(1)

    # checkpoint_path = "/alpha/my_marl/results/models/qmix_seed87455234_supereasy_2025-01-07 02:58:54.417746"
    # load_step = 9979800
    # params.append('--config=qmix')

    checkpoint_path = "results/models/ippo_seed4_supereasy_2025-01-16 21:06:38.941962"
    load_step = 19770000
    params.append('--config=ippo')
    params.append('--env-config=overcooked2_evaluate')
    
    # Get the defaults from default.yaml
    with open(
        os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r"
    ) as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    alg_config = get_config(params, "--config", "algs")
    env_config = get_config(params, "--env-config", "envs")

    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, alg_config) # 用alg_config更新config_dict
    config_dict = recursive_dict_update(config_dict, env_config)

    config_dict.update({"checkpoint_path": checkpoint_path, 
                        "load_step": load_step})
    
    map_name = config_dict["env_args"]["map_name"]

    # 获取 p0 和 p1 的 agent 类型
    agent_types = config_dict["agents"]
    p0_type = agent_types.get("p0", "random")  # 默认值为 "random"
    p1_type = agent_types.get("p1", "random")  # 默认值为 "random"
    # 根据 agent 类型加载对应的逻辑
    # agent_factories = {
    #     "random": lambda: RandomAgent(),
    #     "rl": lambda: RLAgent(checkpoint_path="results/models/rl_agent_checkpoint"),
    #     "human": lambda: HumanAgent(),
    #     "llm": lambda: LLMAgent(model_path="results/models/llm_model"),
    # }

    # # 初始化 agents
    # agents = {
    #     "p0": agent_factories[p0_type](),
    #     "p1": agent_factories[p1_type](),
    # }

    print(f"Loaded agents: p0={p0_type}, p1={p1_type}")


    print("config_dict:",config_dict)
    # now add all the config to sacred
    ex.add_config(config_dict)

    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(
        results_path, f"sacred/{config_dict['name']}/{map_name}"
    )

    # ex.observers.append(MongoObserver(db_name="marlbench")) #url='172.31.5.187:27017'))
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    # ex.observers.append(MongoObserver())

    ex.run_commandline(params)
