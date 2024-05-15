from typing import Tuple

import numpy as np

from python_code import conf
from python_code.datasets.channels.sed_channel import SEDChannel
from python_code.utils.constants import Phase


class Transmitter:
    def __init__(self, phase: Phase):
        self.phase = phase

    def transmit(self, s: np.ndarray, index: int, users: int) -> Tuple[np.ndarray, np.ndarray]:
        H = SEDChannel.get_channel_matrix(conf.n_ant, users)
        snrs = SEDChannel.get_snrs(users, index, self.phase)
        # pass through datasets
        rx = SEDChannel.transmit(s=s, h=H, snrs=snrs)
        return rx, np.concatenate([H, snrs.reshape(-1, 1)], axis=1)
