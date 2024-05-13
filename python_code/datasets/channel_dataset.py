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

    def __init__(self, block_length: int, blocks_num: int, pilots_length: int):
        self.block_length = block_length
        self.blocks_num = blocks_num
        self.generator = Generator(block_length, pilots_length)
        self.modulator = BPSKModulator()
        self.transmitter = Transmitter()

    def get_data(self, phase, database: list):
        if database is None:
            database = []
        mx_full = np.empty((self.blocks_num, self.block_length, conf.n_user))
        rx_full = np.empty((self.blocks_num, self.block_length, conf.n_ant))
        hs = []
        # accumulate words until reaches desired number
        for index in range(self.blocks_num):
            mx = self.generator.generate()
            tx = self.modulator.modulate(mx)
            rx, h = self.transmitter.transmit(tx, index, phase)
            # accumulate
            mx_full[index] = mx
            rx_full[index] = rx
            hs.append(h)

        database.append((mx_full, rx_full, hs))

    def __getitem__(self, phase: Phase) -> Tuple[torch.Tensor, torch.Tensor, List[List[float]]]:
        database = []
        # do not change max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.get_data, phase, database)
        mx, rx, hs = (arrays for arrays in zip(*database))
        mx, rx = torch.Tensor(np.concatenate(mx)).to(device=DEVICE), torch.from_numpy(np.concatenate(rx)).to(device=DEVICE)
        return mx, rx, hs[0]
