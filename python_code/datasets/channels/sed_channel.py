import numpy as np

from python_code import conf
from python_code.utils.constants import Phase

SNR = 8
COEF = 3
# per user: (Min SNR, Max SNR, Number of blocks between peaks)
TRAIN_SNR_PER_USER = [(SNR, COEF * SNR, 5), (SNR, COEF * SNR, 10), (SNR, COEF * SNR, 15), (SNR, COEF * SNR, 20),
                      (SNR, COEF * SNR, 25), (SNR, COEF * SNR, 30), (SNR, COEF * SNR, 35), (SNR, COEF * SNR, 40),
                      (SNR, COEF * SNR, 13), (SNR, COEF * SNR, 25), (SNR, COEF * SNR, 7), (SNR, COEF * SNR, 21) ]
SNR2 = 8
COEF = 3
TEST_SNR_PER_USER = [(SNR2, COEF * SNR2, 3), (SNR2, COEF * SNR2, 10), (SNR2, COEF * SNR2, 7), (SNR2, COEF * SNR2, 12),
                     (SNR2, COEF * SNR2, 15), (SNR2, COEF * SNR2, 22), (SNR2, COEF * SNR2, 23), (SNR2, COEF * SNR2, 14),
                     (SNR2, COEF * SNR2, 25), (SNR2, COEF * SNR2, 18), (SNR2, COEF * SNR2, 30), (SNR2, COEF * SNR2, 11)]

SNR_PER_USER_DICT = {Phase.TRAIN: TRAIN_SNR_PER_USER, Phase.TEST: TEST_SNR_PER_USER}


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
            min_snr, max_snr, peak_blocks = SNR_PER_USER_DICT[phase][i]
            cos_val = np.cos(np.pi * index / peak_blocks)
            first_term = (1 - cos_val) * min_snr / 2
            second_term = (1 + cos_val) * max_snr / 2
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
        snrs = 10 ** (snrs / 20)
        snrs_mat = np.eye(h.shape[0])
        for i in range(len(snrs_mat)):
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
