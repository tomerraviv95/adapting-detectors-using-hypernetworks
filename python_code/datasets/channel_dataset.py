import concurrent.futures
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from python_code import DEVICE, conf
from python_code.datasets.communications_blocks.generator import Generator
from python_code.datasets.communications_blocks.modulator import BPSKModulator
from python_code.datasets.communications_blocks.transmitter import Transmitter
from python_code.utils.constants import Phase


class ChannelModelDataset(Dataset):
    """
    Dataset object for the datasets. Used in training and evaluation.
    Returns (transmitted, received, channel_coefficients) batch.
    """

    def __init__(self):
        self.blocks_num = conf.blocks_num
        self.generator = Generator()
        self.modulator = BPSKModulator()
        self.transmitter = Transmitter()

    def get_data(self, phase, database: list):
        if database is None:
            database = []
        mx_full = np.empty((self.blocks_num, conf.block_length, conf.n_user))
        rx_full = np.empty((self.blocks_num, conf.block_length, conf.n_ant))
        snrs_list = []
        # accumulate words until reaches desired number
        for index in range(conf.blocks_num):
            mx = self.generator.generate()
            tx = self.modulator.modulate(mx)
            rx, snrs = self.transmitter.transmit(tx, index, phase)
            # accumulate
            mx_full[index] = mx
            rx_full[index] = rx
            snrs_list.append(snrs)

        database.append((mx_full, rx_full, snrs_list))

    def __getitem__(self, phase: Phase) -> Tuple[torch.Tensor, torch.Tensor, List[List[float]]]:
        database = []
        # do not change max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.get_data, phase, database)
        mx, rx, snrs_list = (np.concatenate(arrays) for arrays in zip(*database))
        mx, rx = torch.Tensor(mx).to(device=DEVICE), torch.from_numpy(rx).to(device=DEVICE)
        return mx, rx, snrs_list
