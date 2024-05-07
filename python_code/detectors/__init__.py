from python_code.utils.constants import DetectorType

from python_code.detectors.seq_deepsic.seq_deep_sic_trainer import SeqDeepSICTrainer

DETECTORS_TYPE_DICT = {DetectorType.seq_model.name: SeqDeepSICTrainer}