from python_code.detectors.hypernetwork_deepsic.hypernetwork_deepsic_trainer import HypernetworkDeepSICTrainer
from python_code.detectors.recurrent_deepsic.recurrent_deepsic_trainer import RecDeepSICTrainer
from python_code.detectors.rnn_hypernetwork_deepsic.rnn_hypernetwork_deepsic_trainer import \
    RNNHypernetworkDeepSICTrainer
from python_code.detectors.seq_deepsic.seq_deep_sic_trainer import SeqDeepSICTrainer
from python_code.utils.constants import DetectorType

DETECTORS_TYPE_DICT = {DetectorType.seq_deepsic.name: SeqDeepSICTrainer,
                       DetectorType.rec_deepsic.name: RecDeepSICTrainer,
                       DetectorType.hyper_deepsic.name: HypernetworkDeepSICTrainer,
                       DetectorType.rnn_hyper_deepsic.name: RNNHypernetworkDeepSICTrainer}
