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
from typing import Dict, Tuple, List
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
    # params.append('--config=mappo')
    # params.append('--env-config=overcooked2')
    config_dict = {}
    # Load algorithm and env base configs
    alg_config = get_config(params, "--config", "algs")
    env_config = get_config(params, "--env-config", "envs")
    
    # 参数优先级 alg > env > default
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config) # 用alg_config覆盖config_dict

    map_name = config_dict["env_args"]["map_name"]

    print("config_dict:",config_dict)
    
    # now add all the config to sacred
    ex.add_config(config_dict)

    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(
        results_path, f"sacred/{config_dict['name']}/{map_name}"
    )

    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
