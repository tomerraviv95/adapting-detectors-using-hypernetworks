from typing import List

import torch
from torch import nn

from python_code import DEVICE
from python_code.detectors.deepsic.deepsic_detector import DeepSICDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import MAX_USERS, DetectorUtil, HALF


class OnlineDeepSICTrainer(Trainer):
    """
    Weights-tied version (across multiple iterations) of the DeepSIC receiver
    """

    def __init__(self):
        self.lr = 1e-3
        self.epochs = 100
        super().__init__()

    def __str__(self):
        return 'Online DeepSIC'

    def _initialize_detector(self):
        # Populate dict of lists of DeepSIC modules. Each key in the dictionary corresponds to a single configuration
        # of K[t] users. The values are the weights, a list of K[t] parameter vectors, one per user.
        self.detector = torch.nn.ModuleDict()
        for user in range(2, MAX_USERS + 1):
            cur_module_list = torch.nn.ModuleList(
                [DeepSICDetector(user, self.hidden_size).to(DEVICE) for _ in range(user)])
            self.detector.update({str(user): torch.nn.ModuleList(cur_module_list)})

    def _soft_symbols_from_probs(self, input: torch.Tensor, user: int, detector_util: DetectorUtil,
                                 i: int = None) -> torch.Tensor:
        return self.softmax(self.detector[str(detector_util.n_users)][user](input.float()))

    def _train_models(self, model: nn.Module, mx_all: List[torch.Tensor],
                      rx_all: List[torch.Tensor],
                      n_user: int):
        for user in range(n_user):
            self.train_model(model[user], mx_all[user], rx_all[user])

    def train(self, mx: torch.Tensor, rx: torch.Tensor, detector_util: DetectorUtil = None):
        """
        Main training function for DeepSIC evaluater. Initializes the probabilities, then propagates them through
        the network, training sequentially each network and not by end-to-end manner (each one individually).
        """
        n_user = mx.shape[1]
        initial_probs = mx.clone()
        tx_all, rx_all = self._prepare_data_for_training(mx, rx, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        self._train_models(self.detector[str(n_user)], tx_all, rx_all, n_user)
        # Initializing the probabilities
        probs_vec = HALF * torch.ones(mx.shape).to(DEVICE)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, self.iterations):
            # Generating soft symbols for training purposes
            detector_util = DetectorUtil(H_hat=None, n_users=n_user)
            probs_vec = self._calculate_posteriors(0, probs_vec, rx, detector_util)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            tx_all, rx_all = self._prepare_data_for_training(mx, rx, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            self._train_models(self.detector[str(n_user)], tx_all, rx_all, n_user)
