from typing import List

import torch

from python_code import DEVICE, conf
from python_code.datasets.communications_blocks.modulator import BPSKModulator
from python_code.detectors.detector_trainer import Detector

EPOCHS = 50


class DeepSICTrainer(Detector):

    def __init__(self):
        self.lr = 1e-3
        self.iterations = 3
        super().__init__()

    def __str__(self):
        return 'DeepSIC'

    def _initialize_detector(self):
        pass

    def calc_loss(self, est: torch.Tensor, mx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        return self.criterion(input=est, target=mx.long())

    def forward(self, rx: torch.Tensor, snrs_list: List[List[float]] = None) -> torch.Tensor:
        with torch.no_grad():
            # detect and decode
            probs_vec = 0.5 * torch.ones([rx.shape[0], conf.n_user]).to(DEVICE).float()
            for i in range(self.iterations):
                probs_vec = self.calculate_posteriors(i + 1, probs_vec, rx, snrs_list)
            detected_words = self.symbols_from_prob(probs_vec)
            return detected_words

    def symbols_from_prob(self, probs_vec: torch.Tensor) -> torch.Tensor:
        symbols_word = torch.sign(probs_vec - 0.5)
        detected_words = BPSKModulator.demodulate(symbols_word)
        return detected_words

    def prepare_data_for_training(self, tx: torch.Tensor, rx: torch.Tensor, probs_vec: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
        """
        Generates the data for each user
        """
        tx_all = []
        rx_all = []
        for k in range(conf.n_user):
            idx = [user_i for user_i in range(conf.n_user) if user_i != k]
            current_y_train = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            tx_all.append(tx[:, k])
            rx_all.append(current_y_train)
        return tx_all, rx_all

    def calculate_posteriors(self, i: int, probs_vec: torch.Tensor, rx: torch.Tensor, snrs_list=None) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks for a single iteration.
        """
        next_probs_vec = torch.zeros(probs_vec.shape).to(DEVICE)
        for user in range(conf.n_user):
            idx = [user_i for user_i in range(conf.n_user) if user_i != user]
            input = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            with torch.no_grad():
                output = self.soft_symbols_from_probs(i, input, user, snrs_list)
            next_probs_vec[:, user] = output[:, 1:].reshape(next_probs_vec[:, user].shape)
        return next_probs_vec
