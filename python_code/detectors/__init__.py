from python_code.detectors.hypernetwork_deepsic.hypernetwork_deepsic_trainer import HypernetworkDeepSICTrainer
from python_code.detectors.recurrent_deepsic.recurrent_deepsic_trainer import RecDeepSICTrainer
from python_code.utils.constants import DetectorType

DETECTORS_TYPE_DICT = {DetectorType.rec_deepsic.name: RecDeepSICTrainer,
                       DetectorType.hyper_deepsic.name: HypernetworkDeepSICTrainer}
