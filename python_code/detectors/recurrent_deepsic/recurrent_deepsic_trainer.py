from collections import defaultdict
from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.detectors.deepsic_detector import DeepSICDetector
from python_code.detectors.deepsic_trainer import DeepSICTrainer
from python_code.utils.constants import TRAINING_TYPES_DICT, EPOCHS_DICT, MAX_USERS


class RecDeepSICTrainer(DeepSICTrainer):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return TRAINING_TYPES_DICT[conf.training_type].name + ' Recurrent DeepSIC'

    def _initialize_detector(self):
        # populate 1D list for Storing the DeepSIC Networks
        self.detector = {user: [DeepSICDetector(user, self.hidden_size).to(DEVICE) for _ in range(user)] for user in
                         range(2, MAX_USERS)}

    def _soft_symbols_from_probs(self, input: torch.Tensor, user: int, hs: int, i: int = None) -> torch.Tensor:
        return self.softmax(self.detector[hs][user](input.float()))

    def train_model(self, single_model: nn.Module, mx: List[torch.Tensor], rx: List[torch.Tensor]):
        """
        Trains a DeepSIC Network
        """
        self.optimizer = torch.optim.Adam(single_model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        single_model = single_model.to(DEVICE)
        epochs = EPOCHS_DICT[conf.training_type]
        mx = torch.cat(mx)
        rx = torch.cat(rx)
        for _ in range(epochs):
            soft_estimation = single_model(rx.float())
            self._run_train_loop(soft_estimation, mx)

    def train_models(self, model: List[DeepSICDetector], mx_all: List[List[torch.Tensor]],
                     rx_all: List[List[torch.Tensor]],
                     n_user):
        for user in range(n_user):
            self.train_model(model[user], [mx_all[i][user] for i in range(len(mx_all))],
                             [rx_all[i][user] for i in range(len(rx_all))])

    def train(self, mxs: torch.Tensor, rxs: torch.Tensor, snrs_list=None):
        """
        Main training function for DeepSIC evaluater. Initializes the probabilities, then propagates them through
        the network, training sequentially each network and not by end-to-end manner (each one individually).
        """
        # Obtaining the data for each number of users
        mx_all_by_user, rx_all_by_user = defaultdict(list), defaultdict(list)
        for mx, rx in zip(mxs, rxs):
            n_user = mx.shape[1]
            probs_vec = torch.rand(mx.shape).to(DEVICE)
            mx_cur, rx_cur = self._prepare_data_for_training(mx, rx, probs_vec)
            mx_all_by_user[n_user].append(mx_cur)
            rx_all_by_user[n_user].append(rx_cur)

        for n_user in mx_all_by_user.keys():
            # Training the DeepSIC networks for the iteration>1
            self.train_models(self.detector[n_user], mx_all_by_user[n_user], rx_all_by_user[n_user], n_user)
