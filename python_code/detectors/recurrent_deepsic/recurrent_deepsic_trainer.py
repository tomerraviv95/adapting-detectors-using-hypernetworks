from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.detectors.deepsic_detector import DeepSICDetector
from python_code.detectors.deepsic_trainer import DeepSICTrainer, EPOCHS
from python_code.utils.constants import TRAINING_TYPES_DICT


class RecDeepSICTrainer(DeepSICTrainer):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return TRAINING_TYPES_DICT[conf.training_type] + ' Recurrent DeepSIC'

    def _initialize_detector(self):
        # populate 1D list for Storing the DeepSIC Networks
        self.detector = [DeepSICDetector() for _ in range(conf.n_user)]

    def _soft_symbols_from_probs(self, input: torch.Tensor, user: int, i=None, snrs_list=None) -> torch.Tensor:
        return self.softmax(self.detector[user](input.float()))

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
            self._run_train_loop(soft_estimation, mx)

    def train_models(self, model: List[DeepSICDetector], tx_all: List[torch.Tensor], rx_all: List[torch.Tensor]):
        for user in range(conf.n_user):
            self.train_model(model[user], tx_all[user], rx_all[user])

    def train(self, mx: torch.Tensor, rx: torch.Tensor, snrs_list=None):
        """
        Main training function for DeepSIC evaluater. Initializes the probabilities, then propagates them through
        the network, training sequentially each network and not by end-to-end manner (each one individually).
        """
        if len(mx.shape) == 3:
            mx = mx.reshape(-1, mx.shape[2])
            rx = rx.reshape(-1, rx.shape[2])
        # Initializing the probabilities
        probs_vec = 0.5 * torch.ones(mx.shape).to(DEVICE)
        # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
        mx_all, rx_all = self._prepare_data_for_training(mx, rx, probs_vec)
        # Training the DeepSIC networks for the iteration>1
        self.train_models(self.detector, mx_all, rx_all)
