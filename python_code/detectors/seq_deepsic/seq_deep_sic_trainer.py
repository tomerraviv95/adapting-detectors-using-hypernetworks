from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.detectors.deepsic_detector import DeepSICDetector
from python_code.detectors.deepsic_trainer import DeepSICTrainer
from python_code.utils.constants import TRAINING_TYPES_DICT, MAX_USERS

EPOCHS = 50


class SeqDeepSICTrainer(DeepSICTrainer):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return TRAINING_TYPES_DICT[conf.training_type].name + ' Sequential DeepSIC'

    def _initialize_detector(self):
        self.detector = {user: [DeepSICDetector(user, self.hidden_size).to(DEVICE) for _ in range(user)] for user in
                         range(2, MAX_USERS + 1)}

    def _soft_symbols_from_probs(self, input: torch.Tensor, user: int, hs: int, i: int = None) -> torch.Tensor:
        return self.softmax(self.detector[hs][user](input.float()))

    def _train_model(self, single_model: nn.Module, mx: torch.Tensor, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        self.optimizer = torch.optim.Adam(single_model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        single_model = single_model.to(DEVICE)
        y_total = rx.float()
        for _ in range(EPOCHS):
            soft_estimation = single_model(y_total)
            self._run_train_loop(soft_estimation, mx)

    def _train_models(self, model: List[List[DeepSICDetector]], i: int, mx_all: List[torch.Tensor],
                      rx_all: List[torch.Tensor], n_user):
        for user in range(n_user):
            self._train_model(model[i], mx_all[user], rx_all[user])

    def train(self, mx: torch.Tensor, rx: torch.Tensor, snrs_list=None):
        """
        Main training function for DeepSIC evaluater. Initializes the probabilities, then propagates them through
        the network, training sequentially each network and not by end-to-end manner (each one individually).
        """
        mx = mx[0]
        rx = rx[0]
        n_user = mx.shape[1]
        # Initializing the probabilities
        probs_vec = 0.5 * torch.ones(mx.shape).to(DEVICE)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(self.iterations):
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            mx_all, rx_all = self._prepare_data_for_training(mx, rx, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self._train_models(self.detector[n_user], i, mx_all, rx_all, n_user)
            # Generating soft symbols for training purposes
            probs_vec = self._calculate_posteriors(i + 1, probs_vec, rx)
