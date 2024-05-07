import numpy as np

from python_code import conf

H_COEF = 0.8
MAX_SNR_PER_USER = [16, 8, 12, 14, 6, 8, 10, 9, 4, 6, 15, 16]  # Max SNR per user in dB
MIN_SNR_PER_USER = [4, 6, 4, 2, 5, 3, 2, 5, 6, 2, 5, 6]  # Min SNR per user in dB
TIME_BETWEEN_PEAKS = [10, 5, 13, 20, 8, 4, 3, 10, 10, 9, 13, 12]  # Number of blocks between MAX and MIN snrs


class SEDChannel:
    @staticmethod
    def get_channel_matrix(n_ant: int, n_user: int) -> np.ndarray:
        H_row = np.array([i for i in range(n_ant)])
        H_row = np.tile(H_row, [n_user, 1]).T
        H_column = np.array([i for i in range(n_user)])
        H_column = np.tile(H_column, [n_ant, 1])
        H = np.exp(-np.abs(H_row - H_column))
        return H

    @staticmethod
    def get_snrs(n_user: int, index: int) -> np.ndarray:
        snrs = []
        for i in range(n_user):
            # oscillating snr between MIN and MAX SNRs
            # f(-1) = Min, f(1) = Max
            # f(x) = (1-x) * Min/2 + (1+x) * Max/2
            cos_val = np.cos(np.pi * index / TIME_BETWEEN_PEAKS[i])
            first_term = (1 - cos_val) * MIN_SNR_PER_USER[i] / 2
            second_term = (1 + cos_val) * MAX_SNR_PER_USER[i] / 2
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
        snrs = (10 ** (snrs / 20)).reshape(-1, 1)
        snrs_mat = np.tile(snrs, [1, conf.n_ant])
        snrs_scaled_h = np.matmul(snrs_mat, h)
        conv = SEDChannel._compute_channel_signal_convolution(snrs_scaled_h, s)
        w = np.random.randn(conf.n_ant, s.shape[1])
        y = conv + w
        return y

    @staticmethod
    def _compute_channel_signal_convolution(h: np.ndarray, s: np.ndarray) -> np.ndarray:
        conv = np.matmul(h, s)
        return conv
