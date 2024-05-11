import numpy as np

from python_code import conf
from python_code.datasets.channels.sed_channel import SEDChannel


class Transmitter:
    def transmit(self, s: np.ndarray, index: int,phase) -> np.ndarray:
        H = SEDChannel.get_channel_matrix(conf.n_ant, conf.n_user)
        snrs = SEDChannel.get_snrs(conf.n_user, index,phase)
        # pass through datasets
        rx, snrs_scaled_h = SEDChannel.transmit(s=s, h=H, snrs=snrs)
        return rx, snrs_scaled_h
