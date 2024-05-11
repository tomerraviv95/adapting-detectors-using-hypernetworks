import random
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
from python_code.utils.constants import TRAINING_TYPES_DICT, TrainingType

EPOCHS = 20


class HypernetworkDeepSICTrainer(DeepSICTrainer):

    def __init__(self):
        super().__init__()
        self.train_context_embedding = []
        self.test_context_embedding = []
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
        user_embeddings = self.user_embedder((torch.Tensor(snrs).to(DEVICE).reshape(-1, 1)))
        context_embedding = torch.zeros_like(user_embeddings[0]).to(DEVICE)
        for j in range(conf.n_user):
            context_embedding += (-1) ** (j == user) * user_embeddings[j]
        return context_embedding

    def train(self, message_words: torch.Tensor, received_words: torch.Tensor, snrs_list: List[List[float]]):
        total_parameters = self.user_embedder.parameters()
        for user in range(conf.n_user):
            total_parameters = chain(total_parameters, self.hypernetworks[user].parameters())
        self.optimizer = torch.optim.Adam(total_parameters, lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mse_criterion = torch.nn.MSELoss()
        for epoch in range(EPOCHS):
            curr_batch = np.random.choice(len(snrs_list), 50)
            for i in curr_batch:
                loss = 0
                for user in range(conf.n_user):
                    mx, rx, snrs = message_words[i], received_words[i], snrs_list[i]
                    # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                    probs_vec = 0.5 * torch.ones(mx.shape).to(DEVICE)
                    mx_all, rx_all = self._prepare_data_for_training(mx, rx, probs_vec)
                    additivity_loss = self.additivity_loss(snrs)
                    with torch.no_grad():
                        # get the context embedding for the hypernetwork based on the user and snrs
                        context_embedding = self._get_context_embedding(snrs, user)
                        self.train_context_embedding.append(context_embedding.detach().cpu().numpy())
                    # Forward pass through the hypernetwork to generate weights
                    weights = self.hypernetworks[user](context_embedding)
                    # Set the generated weights to the base network in the forward pass of deepsic
                    soft_estimation = self.hyper_deepsic(rx_all[user].float(), weights)
                    # calculate loss
                    ce_loss = self._calc_loss(est=soft_estimation, mx=mx_all[user])
                    loss += additivity_loss
                    loss += ce_loss
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        snrs = [3, 4, 5, 1]
        user_embeddings = self.user_embedder((torch.Tensor(snrs).to(DEVICE).reshape(-1, 1)))
        sum_user_embeddings = torch.sum(user_embeddings, dim=0)
        target_embeddings = self.user_embedder((torch.Tensor(list(range(21))).to(DEVICE).reshape(-1, 1)))
        for snr, target_embedding in enumerate(target_embeddings):
            print(snr, torch.linalg.norm(target_embedding - sum_user_embeddings))

    def additivity_loss(self, snrs: List[float]):
        coefs = [(-1) ** random.randint(0, 1) for _ in range(len(snrs))]
        user_embeddings = self.user_embedder((torch.Tensor(snrs).to(DEVICE).reshape(-1, 1)))
        sum_user_embeddings = 0
        target_snr = 0
        for coef, user_embedding, snr in zip(coefs, user_embeddings, snrs):
            sum_user_embeddings += coef * user_embedding
            target_snr += coef * snr
        target_snr = torch.Tensor([target_snr]).to(DEVICE).reshape(-1, 1)
        target = self.user_embedder((target_snr))
        loss = self.mse_criterion(input=sum_user_embeddings, target=target)
        return loss
