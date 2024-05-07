import random
from collections import namedtuple

import numpy as np
import torch

from python_code import conf
from python_code.datasets.channel_dataset import ChannelModelDataset

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
        self._initialize_dataloader()

    def _initialize_dataloader(self):
        """
        Sets up the data loader - a generator from which we draw batches of MIMO symbols and their corresponding
        channel observations, in iterations
        """
        self.channel_dataset = ChannelModelDataset()

    def evaluate(self) -> MetricOutput:
        """
        The online evaluation run.
        Main function for running the experiments of sequential transmission of pilots and
        data for the paper.
        :return: list of ber per timestep
        """
        # print(f"Detecting using {str(self.detector)}")
        torch.cuda.empty_cache()
        ser_list, ber_list, ece_list = [], [], []
        # draw words for a given snr
        message_words, transmitted_words, received_words = self.channel_dataset.__getitem__(snr_list=[conf.snr])
        # detect sequentially
        for block_ind in range(conf.blocks_num):
            print('*' * 20)
            print(f'current: {block_ind}')
            # get current word and datasets
            mx, tx, rx = message_words[block_ind], transmitted_words[block_ind], received_words[block_ind]
            # ser, _ = calculate_error_rate(detected_symbols_words, tx_data)
            # ser_list.append(ser)
            # print(f'symbol error rate: {ser}')
        metrics_output = MetricOutput(ser_list=ser_list)
        print(f'Avg SER:{sum(metrics_output.ser_list) / len(metrics_output.ser_list)}')
        return metrics_output


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate()
