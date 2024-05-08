from enum import Enum


class DetectorType(Enum):
    seq_deepsic = 'seq_deepsic'
    rec_deepsic = 'rec_deepsic'
    online_hyper_deepsic = 'online_hyper_deepsic'

class Phase(Enum):
    TRAIN = 'TRAIN'
    TEST = 'TEST'
