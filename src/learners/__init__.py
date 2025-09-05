from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .ppo_learner import PPOLearner
from .policy_gradient_v2 import PGLearner_v2
from .nq_learner import NQLearner

REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["policy_gradient_v2"] = PGLearner_v2
REGISTRY["nq_learner"] = NQLearner
