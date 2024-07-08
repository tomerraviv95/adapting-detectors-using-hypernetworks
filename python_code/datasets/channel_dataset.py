import concurrent.futures
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from python_code import DEVICE, conf
from python_code.datasets.communications_blocks.generator import Generator
from python_code.datasets.communications_blocks.modulator import BPSKModulator
from python_code.datasets.communications_blocks.transmitter import Transmitter
from python_code.datasets.communications_blocks.users_network import UsersNetwork
from python_code.utils.constants import Phase


class ChannelModelDataset(Dataset):
    """
    Dataset object for the training and test generation. Used in training and evaluation.
    Returns (transmitted, received) batch.
    """

    def __init__(self, block_length: int, blocks_num: int, pilots_length: int, phase: Phase):
        self.block_length = block_length
        self.blocks_num = blocks_num
        self.users_network = UsersNetwork(phase)
        self.generator = Generator(block_length, pilots_length)
        self.modulator = BPSKModulator()
        self.transmitter = Transmitter(phase)

    def get_data(self, database: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        if database is None:
            database = []
        mx_full = []
        rx_full = np.empty((self.blocks_num, self.block_length, conf.n_ant))
        # accumulate pairs of transmitted and received words
        for index in range(self.blocks_num):
            users = self.users_network.get_current_users(index)
            mx = self.generator.generate(users)
            tx = self.modulator.modulate(mx)
            rx = self.transmitter.transmit(tx, index, users)
            # accumulate
            mx_full.append(mx)
            rx_full[index] = rx
        database.append((mx_full, rx_full))

    def __getitem__(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        database = []
        # do not change max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(self.get_data, database)
        mx, rx = (arrays for arrays in zip(*database))
        rx = torch.from_numpy(np.concatenate(rx)).to(device=DEVICE)
        mx = [torch.Tensor(mx[0][i]).to(device=DEVICE) for i in range(len(mx[0]))]
        return mx, rx
