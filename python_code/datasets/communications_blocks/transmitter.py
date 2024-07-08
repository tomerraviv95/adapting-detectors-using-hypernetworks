import numpy as np

from python_code import conf
from python_code.datasets.channels.cost_channel import COSTChannel
from python_code.datasets.channels.sed_channel import SEDChannel
from python_code.utils.constants import Phase, ChannelType

CHANNELS_DICT = {ChannelType.SED.name: SEDChannel,
                 ChannelType.COST.name: COSTChannel}


class Transmitter:
    def __init__(self, phase: Phase):
        self.phase = phase
        self.train_test_mismatch = (self.phase == Phase.TRAIN) and (conf.train_test_mismatch)

    def transmit(self, s: np.ndarray, index: int, users: int) -> np.ndarray:
        if self.train_test_mismatch:
            # if mismatch is True, set the training dataset to one opposite to the test channel
            cur_channel = COSTChannel if conf.channel_type == ChannelType.SED.name else SEDChannel
        else:
            # otherwise not mismatch, the train and test channels match
            cur_channel = CHANNELS_DICT[conf.channel_type]
        snrs_db = cur_channel.get_snrs(users, index, self.phase)
        H = cur_channel.get_channel_matrix(conf.n_ant, users, index, self.phase)
        # pass through datasets
        rx = cur_channel.transmit(s=s, H=H, snrs_db=snrs_db)
        return rx
