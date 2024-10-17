from python_code.detectors.deepsic.joint_deepsic import JointDeepSICTrainer
from python_code.detectors.deepsic.online_deepsic import OnlineDeepSICTrainer
from python_code.detectors.hypernetwork_deepsic.hypernetwork_deepsic_trainer import HypernetworkTrainer
from python_code.detectors.icl_detector import ICLDetector
from python_code.utils.constants import DetectorType

DETECTORS_TYPE_DICT = {DetectorType.online_deepsic.name: OnlineDeepSICTrainer,
                       DetectorType.joint_deepsic.name: JointDeepSICTrainer,
                       DetectorType.hyper_deepsic.name: HypernetworkTrainer,
                       DetectorType.icl_detector.name: ICLDetector}
