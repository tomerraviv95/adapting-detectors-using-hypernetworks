from python_code.detectors.recurrent_deepsic.recurrent_deepsic_trainer import RecDeepSICTrainer
from python_code.detectors.seq_deepsic.seq_deep_sic_trainer import SeqDeepSICTrainer
from python_code.utils.constants import DetectorType

DETECTORS_TYPE_DICT = {DetectorType.seq_model.name: SeqDeepSICTrainer,
                       DetectorType.rec_deepsic_model.name: RecDeepSICTrainer}
