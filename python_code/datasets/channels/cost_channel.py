import os

import numpy as np
import scipy.io

from dir_definitions import COST2100_TRAIN_DIR, COST2100_TEST_DIR
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase, TRAINING_BLOCKS_PER_CONFIG

conf = Config()

SCALING_COEF = 0.4
MAX_FRAMES = 25

SNR = 10
MIN_POWER, MAX_POWER = -70, -20


class COSTChannel:

    @staticmethod
    def get_snrs(n_user: int, index: int, phase: Phase) -> np.ndarray:
        snrs = [SNR for _ in range(n_user)]
        return np.array(snrs)

    @staticmethod
    def get_channel_matrix(n_ant: int, n_user: int, index: int, phase: Phase) -> np.ndarray:
        total_h = np.empty([n_user, n_ant])
        for i in range(1, n_ant + 1):
            if phase == Phase.TRAIN:
                channel = 1 + index // TRAINING_BLOCKS_PER_CONFIG
                path_to_ant_mat = os.path.join(COST2100_TRAIN_DIR, f'channel_{channel}_ant_{i}.mat')
            else:
                path_to_ant_mat = os.path.join(COST2100_TEST_DIR, f'channel_0_ant_{i}.mat')
            total_h_user = scipy.io.loadmat(path_to_ant_mat)['h_omni_power']
            norm_h_user = (total_h_user - MIN_POWER) / (MAX_POWER - MIN_POWER)
            cur_h_user = norm_h_user[:n_user, index % TRAINING_BLOCKS_PER_CONFIG]
            total_h[:, i - 1] = SCALING_COEF * cur_h_user  # beamforming to reduce sidelobes
        # applying beamforming for each user
        total_h[np.arange(n_user), np.arange(n_user)] = 1
        if np.any(total_h < 0) or np.any(total_h > 1):
            print('Error in the normalization of the channel! values either smaller than 0 or larger than 1')
            raise ValueError('Fail')
        return total_h

    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snrs: np.ndarray) -> np.ndarray:
        """
        The MIMO COST2100 Channel
        :param s: to transmit symbol words
        :param snr: signal-to-noise value
        :param h: channel coefficients
        :return: received word
        """
        snrs = 10 ** (snrs / 20)
        snrs_mat = np.eye(h.shape[0])
        for i in range(len(snrs_mat)):
            snrs_mat[i, i] = snrs[i]
        # Users X antennas matrix. Scale each row by the TRAIN_SNR of the given user.
        snrs_scaled_h = np.matmul(snrs_mat, h)
        conv = np.matmul(s, snrs_scaled_h)
        w = np.random.randn(s.shape[0], conf.n_ant)
        y = conv + w
        return y
