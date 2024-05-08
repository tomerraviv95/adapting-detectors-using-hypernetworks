from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.detectors.deepsic_detector import DeepSICDetector
from python_code.detectors.deepsic_trainer import DeepSICTrainer
from python_code.detectors.hypernetwork_deepsic.hyper_deepsic import HyperDeepSICDetector
from python_code.detectors.hypernetwork_deepsic.hypernetwork import Hypernetwork
from python_code.detectors.hypernetwork_deepsic.snr_embedder import UserEmbedder

EPOCHS = 20


class OnlineHypernetworkDeepSICTrainer(DeepSICTrainer):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Online Hypernetwork-based DeepSIC'

    def _initialize_detector(self):
        self.base_deepsic = DeepSICDetector()
        total_parameters = [param.numel() for param in self.base_deepsic.parameters()]
        self.hypernetwork = Hypernetwork(32, total_parameters).to(DEVICE)
        self.hyper_deepsic = HyperDeepSICDetector([param.size() for param in self.base_deepsic.parameters()])
        self.inference_weights = [None for _ in range(conf.n_user)]
        self.user_embedder = UserEmbedder(32).to(DEVICE)

    def soft_symbols_from_probs(self, i, input, user, snrs_list):
        if i == 1:
            user_embedding = self.user_embedder(torch.Tensor(snrs_list).to(DEVICE).reshape(-1, 1))
            context_embedding = 0
            for j in range(conf.n_user):
                context_embedding += (-1) ** (j != user) * user_embedding[j]
            self.inference_weights[user] = self.hypernetwork(context_embedding)
        deepsic_output = self.hyper_deepsic(input.float(), self.inference_weights[user])
        return self.softmax(deepsic_output)

    def train_model(self, hypernetwork: nn.Module, mx: torch.Tensor, rx: torch.Tensor, snrs_list, k):
        """
        Trains a hypernetwork DeepSIC Network
        """
        self.optimizer = torch.optim.Adam(hypernetwork.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(EPOCHS):
            user_embedding = self.user_embedder(torch.Tensor(snrs_list).to(DEVICE).reshape(-1, 1))
            context_embedding = 0
            for j in range(conf.n_user):
                context_embedding += (-1) ** (j != k) * user_embedding[j]
            # Forward pass through the hypernetwork to generate weights
            weights = hypernetwork(context_embedding)
            # Set the generated weights to the base network in the forward pass of deepsic
            soft_estimation = self.hyper_deepsic(rx.float(), weights)
            # calculate loss and update parameters of hypernetwork
            loss = self.run_train_loop(soft_estimation, mx)

    def _online_training(self, mx: torch.Tensor, rx: torch.Tensor, snrs_list: List[List[float]]):
        """
        Main training function for DeepSIC evaluater. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """
        if self.train_from_scratch:
            self._initialize_detector()
        # Initializing the probabilities
        probs_vec = 0.5 * torch.ones(mx.shape).to(DEVICE)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(self.iterations):
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            mx_all, rx_all = self.prepare_data_for_training(mx, rx, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            for k in range(conf.n_user):
                self.train_model(self.hypernetwork, mx_all[k], rx_all[k], snrs_list, k)
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(i + 1, probs_vec, rx, snrs_list)
