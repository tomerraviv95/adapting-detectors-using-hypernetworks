import os

import numpy as np
import scipy.io

from dir_definitions import QUADRIGA_TEST_DIR, QUADRIGA_TRAIN_DIR
from python_code.utils.config_singleton import Config
from python_code.utils.constants import Phase

conf = Config()

SCALING_COEF = 0.7


class QuadrigaChannel:

    @staticmethod
    def get_snrs(n_user: int, index: int, phase: Phase) -> np.ndarray:
        snrs = [conf.cost_snr for _ in range(n_user)]
        return np.array(snrs)

    @staticmethod
    def get_channel_matrix(n_ant: int, n_user: int, index: int, phase: Phase) -> np.ndarray:
        # load matrices of size n_user X n_ant X total_frames (100 in this case)
        if phase == Phase.TRAIN:
            # train using different channels
            channel = 1 + index // conf.tasks_number
            path_to_ant_mat = os.path.join(QUADRIGA_TRAIN_DIR, f'channel_magnitudes_{channel}.mat')
        else:
            # test on channel 0
            path_to_ant_mat = os.path.join(QUADRIGA_TEST_DIR, f'channel_magnitudes_0.mat')
        norm_h_user = scipy.io.loadmat(path_to_ant_mat)['magnitudes']
        norm_h_user /= norm_h_user.max()
        # adjust to the current n_users and n_ant configurations, assuming that n_ant >= n_user
        norm_h_user = norm_h_user[:n_user, :n_ant, index % norm_h_user.shape[2]]
        # beamforming (beam focusing) for each user
        norm_h_user[np.arange(n_user), np.arange(n_user)] = 1
        if np.any(norm_h_user < 0) or np.any(norm_h_user > 1):
            print('Error in the normalization of the channel! values out of range')
            raise ValueError('Fail')
        return norm_h_user

    @staticmethod
    def transmit(s: np.ndarray, H: np.ndarray, snrs_db: np.ndarray) -> np.ndarray:
        """
        The MIMO COST2100 Channel
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
