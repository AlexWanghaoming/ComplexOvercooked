from .rnn_agent import RNNAgent
from .conv_agent import ConvAgent
from .n_rnn_agent import NRNNAgent
from .rnn_ppo_agent import RNNPPOAgent
from .mlp_agent import MLPAgent


REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent
REGISTRY["conv_agent"] = ConvAgent

REGISTRY["mlp"] = MLPAgent
