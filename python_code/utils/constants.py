from collections import namedtuple
from enum import Enum

from python_code import conf


class DetectorType(Enum):
    online_deepsic = 'online_deepsic'
    joint_deepsic = 'joint_deepsic'
    hyper_deepsic = 'hyper_deepsic'
    icl_detector = 'icl_detector'


class Phase(Enum):
    TRAIN = 'TRAIN'
    TEST = 'TEST'


class ChannelType(Enum):
    SED = 'SED'
    COST = 'COST'


DetectorUtil = namedtuple("DetectorUtil", "H_hat n_users", defaults=[None, None])

HIDDEN_SIZE = 16
MAX_USERS = conf.n_ant
HALF = 0.5
