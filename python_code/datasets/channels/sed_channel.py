import numpy as np

from python_code import conf
from python_code.utils.constants import Phase

TRAIN_MAX_SNR_PER_USER = [16, 8, 12, 14, 6, 8, 10, 9, 7, 6, 15, 16]  # Max SNR per user in dB
TRAIN_MIN_SNR_PER_USER = [4, 6, 4, 2, 5, 3, 2, 5, 6, 2, 5, 6]  # Min SNR per user in dB
TRAIN_TIME_BETWEEN_PEAKS = [10, 5, 13, 20, 8, 4, 3, 10, 10, 9, 13, 12]  # Number of blocks between MAX and MIN snrs

TEST_MAX_SNR_PER_USER = [12, 18, 6, 10, 11, 17, 20, 9, 9, 6, 15, 16]  # Max SNR per user in dB
TEST_MIN_SNR_PER_USER = [5, 3, 5, 9, 1, 3, 10, 3, 6, 0, 13, 9]  # Min SNR per user in dB
TEST_TIME_BETWEEN_PEAKS = [20, 12, 5, 13, 12, 18, 3, 16, 10, 16, 7, 3]  # Number of blocks between MAX and MIN snrs

MAX_SNR_PER_USER_DICT = {Phase.TRAIN: TRAIN_MAX_SNR_PER_USER, Phase.TEST: TEST_MAX_SNR_PER_USER}
MIN_SNR_PER_USER_DICT = {Phase.TRAIN: TRAIN_MIN_SNR_PER_USER, Phase.TEST: TEST_MIN_SNR_PER_USER}
TIME_BETWEEN_PEAKS_DICT = {Phase.TRAIN: TRAIN_TIME_BETWEEN_PEAKS, Phase.TEST: TEST_TIME_BETWEEN_PEAKS}


class SEDChannel:
    @staticmethod
    def get_channel_matrix(n_ant: int, n_user: int) -> np.ndarray:
        # H is the users X antennas channel matrix
        # H_row has another index of the antenna per location, for each different user
        H_row = np.array([i for i in range(n_ant)])
        H_row = np.tile(H_row, [n_user, 1])
        # H_column has another index of the user per location, for each different antenna
        H_column = np.array([i for i in range(n_user)])
        H_column = np.tile(H_column, [n_ant, 1]).T
        H = np.exp(-np.abs(H_row - H_column))
        return H

    @staticmethod
    def get_snrs(n_user: int, index: int, phase: Phase) -> np.ndarray:
        snrs = []
        for i in range(n_user):
            # oscillating snr between MIN and MAX SNRs
            # f(-1) = Min, f(1) = Max
            # f(x) = (1-x) * Min/2 + (1+x) * Max/2
            cos_val = np.cos(np.pi * index / TIME_BETWEEN_PEAKS_DICT[phase][i])
            first_term = (1 - cos_val) * MIN_SNR_PER_USER_DICT[phase][i] / 2
            second_term = (1 + cos_val) * MAX_SNR_PER_USER_DICT[phase][i] / 2
            cur_snr = first_term + second_term
            snrs.append(cur_snr)
        return np.array(snrs)

    @staticmethod
    def transmit(s: np.ndarray, h: np.ndarray, snrs: np.ndarray) -> np.ndarray:
        """
        The MIMO SED Channel
        :param s: to transmit symbol words
        :param snrs: signal-to-noise value per user
        :param h: channel matrix function
        :return: received word y
        """
        snrs = (10 ** (snrs / 20))
        snrs_mat = np.eye(conf.n_user)
        for i in range(conf.n_user):
            snrs_mat[i, i] = snrs[i]
        # Users X antennas matrix. Scale each row by the SNR of the given user.
        snrs_scaled_h = np.matmul(snrs_mat, h)
        conv = np.matmul(s, snrs_scaled_h)
        w = np.random.randn(s.shape[0], conf.n_ant)
        y = conv + w
        return y

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        conv = np.matmul(h, s)
        return conv
