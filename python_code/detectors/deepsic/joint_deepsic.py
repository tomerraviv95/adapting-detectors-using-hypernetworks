from typing import List

import torch
from torch import nn

from python_code import DEVICE
from python_code.detectors.deepsic.deepsic_detector import DeepSICDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import MAX_USERS, DetectorUtil


class JointDeepSICTrainer(Trainer):
    """
    Weights-tied version (across multiple iterations) of the DeepSIC receiver
    """

    def __init__(self):
        self.lr = 1e-3
        self.epochs = 500
        super().__init__()

    def __str__(self):
        return 'Joint DeepSIC'

    def _initialize_detector(self):
        # Populate dict of lists of DeepSIC modules. Each key in the dictionary corresponds to a single configuration
        # of K[t] users. The values are the weights, a list of K[t] parameter vectors, one per user.
        self.detector = torch.nn.ModuleDict()
        for user in range(2, MAX_USERS + 1):
            cur_module_list = torch.nn.ModuleList(
                [DeepSICDetector(user, self.hidden_size).to(DEVICE) for _ in range(user)])
            self.detector.update({str(user): cur_module_list})

    def _soft_symbols_from_probs(self, input: torch.Tensor, user: int, detector_util: DetectorUtil,
                                 i: int = None) -> torch.Tensor:
        return self.softmax(self.detector[str(detector_util.n_users)][user](input.float()))

    def train_model(self, single_model: nn.Module, mx: List[torch.Tensor], rx: List[torch.Tensor]):
        """
        Trains a DeepSIC Network
        """
        self.optimizer = torch.optim.Adam(single_model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        single_model = single_model.to(DEVICE)
        mx = torch.cat(mx)
        rx = torch.cat(rx)
        for _ in range(self.epochs):
            soft_estimation = single_model(rx.float())
            self._run_train_loop(soft_estimation, mx)

    def _train_models(self, model: nn.Module, mx_all: List[List[torch.Tensor]], rx_all: List[List[torch.Tensor]],
                      n_user: int):
        for user in range(n_user):
            self.train_model(model[user], [mx_all[i][user] for i in range(len(mx_all))],
                             [rx_all[i][user] for i in range(len(rx_all))])

    def train(self, mxs: torch.Tensor, rxs: torch.Tensor, detector_util: DetectorUtil = None):
        """
        Main training function for DeepSIC evaluater. Initializes the probabilities, then propagates them through
        the network, training sequentially each network and not by end-to-end manner (each one individually).
        """
        # training the per-user DeepSIC modules for each configuration of users
        users_indices = [(mx.shape[1], i) for i, mx in enumerate(mxs)]
        for n_user in range(MAX_USERS, 1, -1):
            relevant_ind_pairs = list(filter(lambda x: x[0] == n_user, users_indices))
            if len(relevant_ind_pairs) == 0:
                continue
            print(f"Training modules for {n_user} users")
            tuples = [(mxs[ind[1]], rxs[ind[1]], torch.rand(mxs[ind[1]].shape).to(DEVICE)) for ind in
                      relevant_ind_pairs]
            mx_per_user, rx_per_user = list(zip(*[self._prepare_data_for_training(*cur_tuple) for cur_tuple in tuples]))
            # Training the DeepSIC networks
            self._train_models(self.detector[str(n_user)], mx_per_user, rx_per_user, n_user)
