from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.detectors.deepsic_detector import DeepSICDetector
from python_code.detectors.deepsic_trainer import DeepSICTrainer, EPOCHS


class RecDeepSICTrainer(DeepSICTrainer):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Recurrent DeepSIC'

    def _initialize_detector(self):
        # populate 1D list for Storing the DeepSIC Networks
        self.detector = []
        for _ in range(conf.n_user):
            self.detector.append(DeepSICDetector())

    def soft_symbols_from_probs(self, i, input, user, snrs_list=None):
        output = self.softmax(self.detector[user](input.float()))
        return output

    def joint_training(self, message_words, received_words, snrs_list):
        for tx, rx in zip(message_words, received_words):
            self._online_training(tx, rx)

    def train_model(self, single_model: nn.Module, mx: torch.Tensor, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        self.optimizer = torch.optim.Adam(single_model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        single_model = single_model.to(DEVICE)
        y_total = rx.float()
        for _ in range(EPOCHS):
            soft_estimation = single_model(y_total)
            self.run_train_loop(soft_estimation, mx)

    def train_models(self, model: List[DeepSICDetector], tx_all: List[torch.Tensor],
                     rx_all: List[torch.Tensor]):
        for user in range(conf.n_user):
            self.train_model(model[user], tx_all[user], rx_all[user])

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Main training function for DeepSIC evaluater. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """
        if self.train_from_scratch:
            self._initialize_detector()
        # Initializing the probabilities
        probs_vec = 0.5 * torch.ones(tx.shape).to(DEVICE)
        # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
        tx_all, rx_all = self.prepare_data_for_training(tx, rx, probs_vec)
        # Training the DeepSIC networks for the iteration>1
        self.train_models(self.detector, tx_all, rx_all)
