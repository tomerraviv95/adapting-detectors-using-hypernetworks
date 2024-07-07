from collections import namedtuple
from enum import Enum

from python_code import conf


class DetectorType(Enum):
    seq_deepsic = 'seq_deepsic'
    rec_deepsic = 'rec_deepsic'
    hyper_deepsic = 'hyper_deepsic'


class Phase(Enum):
    TRAIN = 'TRAIN'
    TEST = 'TEST'


class TrainingType(Enum):
    Joint = 'Joint'
    Online = 'Online'


class ChannelType(Enum):
    SED = 'SED'
    COST = 'COST'


DetectorUtil = namedtuple("DetectorUtil", "H_hat n_users", defaults=[None, None])

TRAINING_TYPES_DICT = {'Joint': TrainingType.Joint, 'Online': TrainingType.Online}
HIDDEN_SIZE = 16
MAX_USERS = conf.n_ant
TRAINING_BLOCKS_PER_CONFIG = 100  # Number of training blocks per user number. Only used in offline training.
HALF = 0.5
