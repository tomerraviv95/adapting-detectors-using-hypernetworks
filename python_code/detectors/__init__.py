from python_code.detectors.hypernetwork_deepsic.hypernetwork_deepsic_trainer import HypernetworkTrainer
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer
from python_code.utils.constants import DetectorType

DETECTORS_TYPE_DICT = {DetectorType.deepsic.name: DeepSICTrainer,
                       DetectorType.hyper_deepsic.name: HypernetworkTrainer}
