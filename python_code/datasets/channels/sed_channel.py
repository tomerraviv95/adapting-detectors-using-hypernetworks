import numpy as np

from python_code import conf
from python_code.utils.constants import Phase

TRAIN_SNR = 4
COEF = 3
# per user: (Min TRAIN_SNR, Max TRAIN_SNR, Number of blocks between peaks)
TRAIN_SNR_PER_USER = [(TRAIN_SNR, COEF * TRAIN_SNR, 100), (TRAIN_SNR, COEF * TRAIN_SNR, 99),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 95), (TRAIN_SNR, COEF * TRAIN_SNR, 95),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 96), (TRAIN_SNR, COEF * TRAIN_SNR, 95),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 95), (TRAIN_SNR, COEF * TRAIN_SNR, 8),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 9), (TRAIN_SNR, COEF * TRAIN_SNR, 10),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 11), (TRAIN_SNR, COEF * TRAIN_SNR, 12),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 13), (TRAIN_SNR, COEF * TRAIN_SNR, 14),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 15), (TRAIN_SNR, COEF * TRAIN_SNR, 16),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 17), (TRAIN_SNR, COEF * TRAIN_SNR, 18),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 19), (TRAIN_SNR, COEF * TRAIN_SNR, 20),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 21), (TRAIN_SNR, COEF * TRAIN_SNR, 22),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 23), (TRAIN_SNR, COEF * TRAIN_SNR, 24),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 25), (TRAIN_SNR, COEF * TRAIN_SNR, 70),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 60), (TRAIN_SNR, COEF * TRAIN_SNR, 50),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 40), (TRAIN_SNR, COEF * TRAIN_SNR, 30),
                      (TRAIN_SNR, COEF * TRAIN_SNR, 20), (TRAIN_SNR, COEF * TRAIN_SNR, 10)]
TEST_SNR = 6
COEF = 1.5
TEST_SNR_PER_USER = [(TEST_SNR, COEF * TEST_SNR, 5), (TEST_SNR, COEF * TEST_SNR, 10),
                     (TEST_SNR, COEF * TEST_SNR, 15), (TEST_SNR, COEF * TEST_SNR, 20),
                     (TEST_SNR, COEF * TEST_SNR, 25), (TEST_SNR, COEF * TEST_SNR, 30),
                     (TEST_SNR, COEF * TEST_SNR, 35), (TEST_SNR, COEF * TEST_SNR, 40),
                     (TEST_SNR, COEF * TEST_SNR, 45), (TEST_SNR, COEF * TEST_SNR, 50),
                     (TEST_SNR, COEF * TEST_SNR, 55), (TEST_SNR, COEF * TEST_SNR, 60),
                     (TEST_SNR, COEF * TEST_SNR, 65), (TEST_SNR, COEF * TEST_SNR, 70),
                     (TEST_SNR, COEF * TEST_SNR, 75), (TEST_SNR, COEF * TEST_SNR, 80),
                     (TEST_SNR, COEF * TEST_SNR, 85), (TEST_SNR, COEF * TEST_SNR, 90),
                     (TEST_SNR, COEF * TEST_SNR, 95), (TEST_SNR, COEF * TEST_SNR, 100),
                     (TEST_SNR, COEF * TEST_SNR, 105), (TEST_SNR, COEF * TEST_SNR, 110),
                     (TEST_SNR, COEF * TEST_SNR, 115), (TEST_SNR, COEF * TEST_SNR, 120),
                     (TEST_SNR, COEF * TEST_SNR, 125), (TEST_SNR, COEF * TEST_SNR, 130),
                     (TEST_SNR, COEF * TEST_SNR, 135), (TEST_SNR, COEF * TEST_SNR, 140),
                     (TEST_SNR, COEF * TEST_SNR, 145), (TEST_SNR, COEF * TEST_SNR, 150),
                     (TEST_SNR, COEF * TEST_SNR, 155), (TEST_SNR, COEF * TEST_SNR, 160)]

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
        # Users X antennas matrix. Scale each row by the TRAIN_SNR of the given user.
        snrs_scaled_h = np.matmul(snrs_mat, h)
        conv = np.matmul(s, snrs_scaled_h)
        w = np.random.randn(s.shape[0], conf.n_ant)
        y = conv + w
        return y
