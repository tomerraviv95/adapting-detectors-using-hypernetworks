from itertools import chain
from typing import List

import numpy as np
import torch

from python_code import DEVICE, conf
from python_code.detectors.deepsic_detector import DeepSICDetector
from python_code.detectors.deepsic_trainer import DeepSICTrainer
from python_code.detectors.hypernetwork_deepsic.hyper_deepsic import HyperDeepSICDetector
from python_code.detectors.hypernetwork_deepsic.hypernetwork import Hypernetwork
from python_code.detectors.hypernetwork_deepsic.user_embedder import UserEmbedder, USER_EMB_SIZE
from python_code.utils.constants import TRAINING_TYPES_DICT, TrainingType, HIDDEN_SIZES_DICT

EPOCHS = 15


class HypernetworkDeepSICTrainer(DeepSICTrainer):

    def __init__(self):
        super().__init__()
        self.train_context_embedding = []
        self.test_context_embedding = []
        self.hidden_size = HIDDEN_SIZES_DICT[TrainingType.Online]
        if TRAINING_TYPES_DICT[conf.training_type] == TrainingType.Online:
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

    def _get_context_embedding(self, snrs: List[float], user: int) -> torch.Tensor:
        user_embeddings = self.user_embedder((torch.Tensor(snrs).to(DEVICE)))
        context_embedding = []
        for j in range(conf.n_user):
            if user != j:
                context_embedding.append(user_embeddings[j])
        context_embedding = torch.cat(context_embedding).reshape(1, -1)
        return context_embedding

    def train(self, message_words: torch.Tensor, received_words: torch.Tensor, snrs_list: List[List[float]]):
        total_parameters = self.user_embedder.parameters()
        for user in range(conf.n_user):
            total_parameters = chain(total_parameters, self.hypernetworks[user].parameters())
        self.optimizer = torch.optim.Adam(total_parameters, lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(EPOCHS):
            curr_batch = np.random.choice(len(snrs_list), 10)
            for i in curr_batch:
                for user in range(conf.n_user):
                    mx, rx, snrs = message_words[i], received_words[i], snrs_list[i]
                    # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                    probs_vec = 0.5 * torch.ones(mx.shape).to(DEVICE)
                    mx_all, rx_all = self._prepare_data_for_training(mx, rx, probs_vec)
                    # get the context embedding for the hypernetwork based on the user and snrs
                    context_embedding = self._get_context_embedding(snrs, user)
                    # Forward pass through the hypernetwork to generate weights
                    weights = self.hypernetworks[user](context_embedding)
                    # Set the generated weights to the base network in the forward pass of deepsic
                    soft_estimation = self.hyper_deepsic(rx_all[user].float(), weights)
                    # calculate loss
                    ce_loss = self._calc_loss(est=soft_estimation, mx=mx[:, user])
                    loss = ce_loss
                    # back propagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        with torch.no_grad():
            for i in range(len(snrs_list)):
                for user in range(conf.n_user):
                    mx, rx, snrs = message_words[i], received_words[i], snrs_list[i]
                    # get the context embedding for the hypernetwork based on the user and snrs
                    context_embedding = self._get_context_embedding(snrs, user)
                    self.train_context_embedding.append(context_embedding.detach().cpu().numpy())
