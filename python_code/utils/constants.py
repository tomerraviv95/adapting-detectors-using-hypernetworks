from enum import Enum

from python_code import conf


class DetectorType(Enum):
    seq_deepsic = 'seq_deepsic'
    rec_deepsic = 'rec_deepsic'
    hyper_deepsic = 'hyper_deepsic'
    rnn_hyper_deepsic = 'rnn_hyper_deepsic'


class Phase(Enum):
    TRAIN = 'TRAIN'
    TEST = 'TEST'


class TrainingType(Enum):
    Joint = 'Joint'
    Online = 'Online'


TRAINING_TYPES_DICT = {'Joint': TrainingType.Joint, 'Online': TrainingType.Online}
HIDDEN_SIZES_DICT = {TrainingType.Joint: 16, TrainingType.Online: 16}
EPOCHS = 50

MAX_USERS = conf.n_ant
TRAINING_BLOCKS_PER_CONFIG = 100
USER_EMB_SIZE = conf.n_ant
