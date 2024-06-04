from enum import Enum


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
HIDDEN_SIZES_DICT = {TrainingType.Joint: 64, TrainingType.Online: 16}
EPOCHS = 50

MAX_USERS = 8
TRAINING_SYMBOLS = 10
