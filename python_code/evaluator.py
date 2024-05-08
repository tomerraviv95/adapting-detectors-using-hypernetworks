import random
from collections import namedtuple

import numpy as np
import torch

from python_code import conf
from python_code.datasets.channel_dataset import ChannelModelDataset
from python_code.detectors import DETECTORS_TYPE_DICT
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
        self.train_channel_dataset = ChannelModelDataset()
        self.test_channel_dataset = ChannelModelDataset()

    def evaluate(self) -> MetricOutput:
        """
        The online evaluation run.
        Main function for running the experiments of sequential transmission of pilots and
        data for the paper.
        :return: list of ber per timestep
        """
        print(f"Detecting using {str(self.detector)}")
        torch.cuda.empty_cache()
        ser_list, ber_list, ece_list = [], [], []
        # draw words for a given snr
        message_words, received_words, snrs_list = self.train_channel_dataset.__getitem__()
        TRAIN = True
        if TRAIN:
            self.detector._online_training(message_words, received_words, snrs_list)
        message_words, received_words, snrs_list = self.test_channel_dataset.__getitem__()
        # detect sequentially
        for block_ind in range(conf.blocks_num):
            print('*' * 20)
            print(f'current: {block_ind}')
            # get current word and datasets
            mx, rx = message_words[block_ind], received_words[block_ind]
            mx_pilot, rx_pilot = mx[:conf.pilots_length], rx[:conf.pilots_length]
            mx_data, rx_data = mx[conf.pilots_length:], rx[conf.pilots_length:]
            # run online training on the pilots part
            self.detector._online_training(mx_pilot, rx_pilot, snrs_list[block_ind])
            # detect data part after training on the pilot part
            detected_words = self.detector.forward(rx_data, snrs_list[block_ind])
            ser = calculate_error_rate(detected_words, mx_data)
            ser_list.append(ser)
            print(f'symbol error rate: {ser}')
        metrics_output = MetricOutput(ser_list=ser_list)
        print(f'Avg SER:{sum(metrics_output.ser_list) / len(metrics_output.ser_list)}')
        return metrics_output


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate()
