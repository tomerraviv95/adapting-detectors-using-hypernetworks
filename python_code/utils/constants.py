from enum import Enum


class DetectorType(Enum):
    seq_deepsic = 'seq_deepsic'
    rec_deepsic = 'rec_deepsic'
    hyper_deepsic = 'hyper_deepsic'


class Phase(Enum):
    TRAIN = 'TRAIN'
    TEST = 'TEST'


class TrainingType(Enum):
    joint = 'Joint'
    online = 'Online'


TRAINING_TYPES_DICT = {'joint': TrainingType.joint, 'online': TrainingType.online}
EPOCHS_DICT = {'joint': 50, 'online': 25}
HIDDEN_SIZES_DICT = {TrainingType.joint: 64, TrainingType.online: 16}
