REGISTRY = {}

from .basic_controller import BasicMAC
from .ppo_controller import PPOMAC
from .conv_controller import ConvMAC
from .n_controller import NMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["n_mac"] = NMAC
