import os
import random
from collections import namedtuple

import numpy as np
import torch

from dir_definitions import WEIGHTS_DIR
from python_code import conf
from python_code.datasets.channel_dataset import ChannelModelDataset
from python_code.detectors import DETECTORS_TYPE_DICT
from python_code.utils.channel_estimate import ls_channel_estimation
from python_code.utils.constants import Phase, TrainingType, DetectorType, MAX_USERS, TRAINING_BLOCKS_PER_CONFIG, \
    ChannelType
from python_code.utils.metrics import calculate_error_rate

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)

MetricOutput = namedtuple(
    "MetricOutput",
    "ser_list"
)


class Evaluator(object):
    """
    Implements the evaluation pipeline. Start with initializing the dataloader.
    """

    def __init__(self):
        self.detector = DETECTORS_TYPE_DICT[conf.detector_type]()
        self.test_channel_dataset = ChannelModelDataset(block_length=conf.test_block_length,
                                                        blocks_num=conf.test_blocks_num,
                                                        pilots_length=conf.test_pilots_length,
                                                        phase=Phase.TEST)

        if conf.training_type == 'Joint':
            if not os.path.isdir(WEIGHTS_DIR):
                os.makedirs(WEIGHTS_DIR)
            run_path = self.get_run_path()
            self.weights_path = os.path.join(WEIGHTS_DIR, run_path)
            # try loading weights
            if os.path.exists(self.weights_path):
                checkpoint = torch.load(self.weights_path)
                self.detector.load_state_dict(checkpoint['model_state_dict'])
                return
            # if they don't exist run joint training and save the weights
            train_channel_dataset = ChannelModelDataset(block_length=conf.train_block_length,
                                                        blocks_num=TRAINING_BLOCKS_PER_CONFIG * (MAX_USERS - 1),
                                                        pilots_length=1, phase=Phase.TRAIN)
            message_words, received_words = train_channel_dataset.__getitem__()
            H_hats = [ls_channel_estimation(mx_pilots, rx_pilots) for mx_pilots, rx_pilots in
                      zip(message_words, received_words)]
            self.detector.train(message_words, received_words, H_hats)
            torch.save({'model_state_dict': self.detector.state_dict()}, self.weights_path)

    def get_run_path(self):
        run_name = conf.detector_type + "_" + conf.training_type + "_" + str(conf.n_ant) + "_"
        if (not conf.train_test_mismatch and conf.channel_type == ChannelType.SED.name) or \
                (conf.train_test_mismatch and conf.channel_type == ChannelType.COST.name):
            run_name += ChannelType.SED.name
        else:
            run_name += ChannelType.COST.name + "_" + str(conf.cost_snr)
        run_name += '.pt'
        return run_name

    def evaluate(self) -> MetricOutput:
        """
        The Online evaluation run.
        Main function for running the experiments of sequential transmission of pilots and
        data for the paper.
        :return: list of ber per timestep
        """
        print(f"Detecting using {str(self.detector)}")
        print(self.detector.count_parameters())
        torch.cuda.empty_cache()
        ser_list, ber_list, ece_list = [], [], []
        message_words, received_words = self.test_channel_dataset.__getitem__()
        # detect sequentially
        for block_ind in range(conf.test_blocks_num):
            print('*' * 20)
            print(f'current: {block_ind}')
            # get current word and datasets
            mx, rx = message_words[block_ind], received_words[block_ind]
            mx_pilot, rx_pilot = mx[:conf.test_pilots_length], rx[:conf.test_pilots_length]
            mx_data, rx_data = mx[conf.test_pilots_length:], rx[conf.test_pilots_length:]
            # ---------------------------------------------------------
            # Online training - as in the config "training_type" option
            if self.detector.training_type == TrainingType.Online:
                # run Online training on the pilots part
                self.detector.train([mx_pilot], [rx_pilot])
            if conf.detector_type in [DetectorType.hyper_deepsic.name, DetectorType.rnn_hyper_deepsic.name]:
                # for the hypernetwork calculate the estimation of the channel
                H_hat = ls_channel_estimation(mx_pilot, rx_pilot)
            else:
                # otherwise, no need to calculate the channel coefficients and snr, simply set to the number of users
                H_hat = mx_pilot.shape[1]
            # ---------------------------------------------------------
            # detect data part after training on the pilot part
            detected_words = self.detector.forward(rx_data, H_hat, n_user=mx.shape[1])
            # detected_words = self.detector.forward(rx_data, hs[block_ind], n_user=mx.shape[1])
            ser = calculate_error_rate(detected_words, mx_data)
            ser_list.append(ser)
            print(f'symbol error rate: {ser}')
        metrics_output = MetricOutput(ser_list=ser_list)
        print(f'Avg SER:{sum(metrics_output.ser_list) / len(metrics_output.ser_list)}')
        return metrics_output


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate()
