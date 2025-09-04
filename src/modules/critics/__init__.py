from .coma import COMACritic
from .centralV import CentralVCritic
from .maddpg import MADDPGCritic
from .ac import ACCritic

REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["maddpg_critic"] = MADDPGCritic
REGISTRY["ac_critic"] = ACCritic


def register_pac_critics():
    from .pac_ac import PACCritic

    REGISTRY["pac_critic"] = PACCritic
