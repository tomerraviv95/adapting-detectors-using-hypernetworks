import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.detectors.deepsic_detector import DeepSICDetector
from python_code.detectors.deepsic_trainer import DeepSICTrainer
from python_code.detectors.hypernetwork_deepsic.hyper_deepsic import HyperDeepSICDetector
from python_code.detectors.hypernetwork_deepsic.hypernetwork import Hypernetwork

EPOCHS = 10


class OnlineHypernetworkDeepSICTrainer(DeepSICTrainer):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Online Hypernetwork-based DeepSIC'

    def _initialize_detector(self):
        self.base_deepsic = DeepSICDetector()
        total_parameters = [param.numel() for param in self.base_deepsic.parameters()]
        self.hypernetworks = [Hypernetwork(total_parameters).to(DEVICE) for _ in range(conf.n_user)]
        self.hyper_deepsic = HyperDeepSICDetector([param.size() for param in self.base_deepsic.parameters()])
        self.inference_weights = [None for _ in range(conf.n_user)]

    def soft_symbols_from_probs(self, i, input, user):
        if i == 1:
            self.inference_weights[user] = self.hypernetworks[user](input.float())
        output = []
        for sample_id in range(input.shape[0]):
            cur_weights = [weight[sample_id] for weight in self.inference_weights[user]]
            deepsic_output = self.hyper_deepsic(input[sample_id].float().reshape(1, -1), cur_weights)
            output.append(self.softmax(deepsic_output))
        return torch.cat(output).to(DEVICE)

    def train_model(self, hypernetwork: nn.Module, mx: torch.Tensor, rx: torch.Tensor):
        """
        Trains a hypernetwork DeepSIC Network
        """
        self.optimizer = torch.optim.Adam(hypernetwork.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(EPOCHS):
            soft_estimation = []
            weights = hypernetwork(rx.float())
            for sample_id in range(mx.shape[0]):
                current_input = rx[sample_id].reshape(1, -1).float()
                current_weights = [weight[sample_id] for weight in weights]
                # Forward pass through the hypernetwork to generate weights
                # Set the generated weights to the base network in the forward pass of deepsic
                cur_soft_estimation = self.hyper_deepsic(current_input, current_weights)
                soft_estimation.append(cur_soft_estimation)
            soft_estimation = torch.cat(soft_estimation).to(DEVICE)
            # calculate loss and update parameters of hypernetwork
            self.run_train_loop(soft_estimation, mx)

    def _online_training(self, mx: torch.Tensor, rx: torch.Tensor):
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
            for j in range(conf.n_user):
                self.train_model(self.hypernetworks[j], mx_all[j], rx_all[j])
            # Generating soft symbols for training purposes
            probs_vec = self.calculate_posteriors(i + 1, probs_vec, rx)
