from itertools import chain
from typing import List

import torch

from python_code import DEVICE, conf
from python_code.detectors.deepsic_detector import DeepSICDetector
from python_code.detectors.deepsic_trainer import DeepSICTrainer
from python_code.detectors.hypernetwork_deepsic.hyper_deepsic import HyperDeepSICDetector
from python_code.detectors.hypernetwork_deepsic.hypernetwork import Hypernetwork
from python_code.detectors.hypernetwork_deepsic.user_embedder import UserEmbedder, USER_EMB_SIZE
from python_code.utils.constants import TRAINING_TYPES_DICT, TrainingType, EPOCHS_DICT


class HypernetworkDeepSICTrainer(DeepSICTrainer):

    def __init__(self):
        super().__init__()
        self.train_context_embedding = []
        self.test_context_embedding = []
        if TRAINING_TYPES_DICT[conf.training_type] == TrainingType.online:
            raise ValueError("Online training is not implemented for this detector!!!")

    def __str__(self):
        return TRAINING_TYPES_DICT[conf.training_type].name + ' Hypernetwork-based DeepSIC'

    def _initialize_detector(self):
        self.user_embedder = UserEmbedder().to(DEVICE)
        self.base_deepsic = DeepSICDetector(self.hidden_size)
        total_parameters = [param.numel() for param in self.base_deepsic.parameters()]
        self.hypernetworks = [Hypernetwork(USER_EMB_SIZE, total_parameters).to(DEVICE) for _ in range(conf.n_user)]
        self.hyper_deepsic = HyperDeepSICDetector([param.size() for param in self.base_deepsic.parameters()])
        self.inference_weights = [None for _ in range(conf.n_user)]

    def _soft_symbols_from_probs(self, input: torch.Tensor, user: int, i: int, snrs_list: List[float]) -> torch.Tensor:
        if i == 1:
            context_embedding = self._get_context_embedding(snrs_list, user)
            self.test_context_embedding.append(context_embedding.cpu().numpy())
            self.inference_weights[user] = self.hypernetworks[user](context_embedding)
        deepsic_output = self.hyper_deepsic(input.float(), self.inference_weights[user])
        return self.softmax(deepsic_output)

    def _train_single_hypernetwork(self, message_words: torch.Tensor, received_words: torch.Tensor,
                                   snrs_list: List[List[float]], user: int):
        total_parameters = chain(self.hypernetworks[user].parameters(), self.user_embedder.parameters())
        self.optimizer = torch.optim.Adam(total_parameters, lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        # iterate over the channels
        epochs = EPOCHS_DICT[conf.training_type]
        for _ in range(epochs):
            loss = 0
            for i in range(len(snrs_list)):
                mx, rx, snrs = message_words[i], received_words[i], snrs_list[i]
                # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                probs_vec = 0.5 * torch.ones(mx.shape).to(DEVICE)
                mx_all, rx_all = self._prepare_data_for_training(mx, rx, probs_vec)
                # get the context embedding for the hypernetwork based on the user and snrs
                context_embedding = self._get_context_embedding(snrs, user)
                self.train_context_embedding.append(context_embedding.detach().cpu().numpy())
                # Forward pass through the hypernetwork to generate weights
                weights = self.hypernetworks[user](context_embedding)
                # Set the generated weights to the base network in the forward pass of deepsic
                soft_estimation = self.hyper_deepsic(rx_all[user].float(), weights)
                # calculate loss
                loss += self._calc_loss(est=soft_estimation, mx=mx_all[user])
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _get_context_embedding(self, snrs: List[float], user: int) -> torch.Tensor:
        user_embeddings = self.user_embedder(10 ** (torch.Tensor(snrs).to(DEVICE).reshape(-1, 1) / 20))
        user_embeddings = self.user_embedder(torch.Tensor(snrs).to(DEVICE).reshape(-1, 1))
        context_embedding = torch.zeros_like(user_embeddings[0]).to(DEVICE)
        for j in range(conf.n_user):
            context_embedding += (-1) ** (j != user) * user_embeddings[j]
        # noise_embedding = self.user_embedder(torch.Tensor([1]).to(DEVICE))
        # context_embedding -= noise_embedding
        return context_embedding

    def train(self, mx: torch.Tensor, rx: torch.Tensor, snrs_list: List[List[float]]):
        for user in range(conf.n_user):
            self._train_single_hypernetwork(mx, rx, snrs_list, user)
