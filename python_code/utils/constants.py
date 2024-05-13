from enum import Enum


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


TRAINING_TYPES_DICT = {'Joint': TrainingType.Joint, 'Online': TrainingType.Online}
EPOCHS_DICT = {'Joint': 50, 'Online': 25}
HIDDEN_SIZES_DICT = {TrainingType.Joint: 64, TrainingType.Online: 64}
