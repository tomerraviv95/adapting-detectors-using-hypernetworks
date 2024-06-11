import numpy as np

from python_code import conf
from python_code.datasets.channels.cost_channel import COSTChannel
from python_code.datasets.channels.sed_channel import SEDChannel
from python_code.utils.constants import Phase, ChannelType

TEST_CHANNELS_DICT = {ChannelType.SED.name: SEDChannel,
                      ChannelType.COST.name: COSTChannel}


class Transmitter:
    def __init__(self, phase: Phase):
        self.phase = phase

    def transmit(self, s: np.ndarray, index: int, users: int) -> np.ndarray:
        if self.phase == Phase.TEST:
            cur_channel = TEST_CHANNELS_DICT[conf.channel_type]
        else:
            cur_channel = SEDChannel
        snrs = cur_channel.get_snrs(users, index, self.phase)
        H = cur_channel.get_channel_matrix(conf.n_ant, users, index)
        # pass through datasets
        rx = cur_channel.transmit(s=s, h=H, snrs=snrs)
        return rx
