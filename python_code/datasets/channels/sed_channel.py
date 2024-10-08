import random

import numpy as np

from python_code import conf
from python_code.utils.constants import Phase, MAX_USERS

TRAIN_SNR = 6
COEF = 16 / 6
TRAIN_SNR_PER_USER = [
    (TRAIN_SNR, COEF * TRAIN_SNR, random.randint(10, 100)) for _ in range(MAX_USERS)
]

TEST_SNR = 8
COEF = 12 / 8
TEST_SNR_PER_USER = [
    (TEST_SNR, COEF * TEST_SNR, 5 * (i + 1)) for i in range(MAX_USERS)
]

SNR_PER_USER_DICT = {Phase.TRAIN: TRAIN_SNR_PER_USER, Phase.TEST: TEST_SNR_PER_USER}


class SEDChannel:
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
    def get_channel_matrix(n_ant: int, n_user: int, index: int, phase: Phase = None) -> np.ndarray:
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
    def transmit(s: np.ndarray, H: np.ndarray, snrs_db: np.ndarray) -> np.ndarray:
        """
        The MIMO SED Channel
        :param s: to transmit symbol words
        :param snrs_db: signal-to-noise value per user in db
        :param H: channel coefficients
        :return: received word
        """
        snrs = 10 ** (snrs_db / 20)
        snrs_mat = np.eye(H.shape[0])
        for i in range(len(snrs_mat)):
            snrs_mat[i, i] = snrs[i]
        # Users X antennas matrix. Scale each row by the TRAIN_SNR of the given user.
        snrs_scaled_h = np.matmul(snrs_mat, H)
        conv = np.matmul(s, snrs_scaled_h)
        w = np.random.randn(s.shape[0], conf.n_ant)
        y = conv + w
        return y
