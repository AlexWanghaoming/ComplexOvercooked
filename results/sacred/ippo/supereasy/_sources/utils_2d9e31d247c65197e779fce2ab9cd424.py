import sys
import os
import yaml
from typing import Dict, Tuple, List
try:
    # until python 3.10
    from collections import Mapping
except:
    # from python 3.10
    from collections.abc import Mapping
from copy import deepcopy
import argparse


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int,  default=64)
    parser.add_argument("--use_rnn", type=bool,  default=True)
    parser.add_argument("--n_actions", type=int,  default=6)
    parser.add_argument("--n_agents", type=int,  default=2)
    parser.add_argument("--obs_agent_id", type=bool,  default=True)
    return parser

def get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "config",
                subfolder,
                "{}.yaml".format(config_name),
            ),
            "r",
        ) as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d:Dict, u:Dict) -> Dict:
    """
    update parameters in d using the parameters in u,  recrusively.
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)