from itertools import chain
from typing import List

import numpy as np
import torch
from torch.nn import Embedding

from python_code import DEVICE, conf
from python_code.detectors.deepsic_detector import DeepSICDetector
from python_code.detectors.deepsic_trainer import DeepSICTrainer
from python_code.detectors.hypernetwork_deepsic.hyper_deepsic import HyperDeepSICDetector
from python_code.detectors.hypernetwork_deepsic.hypernetwork import Hypernetwork
from python_code.utils.constants import TRAINING_TYPES_DICT, TrainingType, HIDDEN_SIZES_DICT, MAX_USERS, \
    TRAINING_SYMBOLS, EPOCHS, USER_EMB_SIZE
from python_code.utils.metrics import count_parameters


class HypernetworkDeepSICTrainer(DeepSICTrainer):

    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.train_context_embedding = []
        self.test_context_embedding = []
        if TRAINING_TYPES_DICT[conf.training_type] == TrainingType.Online:
            raise ValueError("Online training is not implemented for this _detector!!!")

    def __str__(self):
        return TRAINING_TYPES_DICT[conf.training_type].name + ' Hypernetwork-based DeepSIC'

    def _initialize_detector(self):
        self.hidden_size = HIDDEN_SIZES_DICT[TrainingType.Online]
        self.base_deepsic = DeepSICDetector(MAX_USERS, self.hidden_size)
        self.no_user_vec = Embedding(1, USER_EMB_SIZE).to(DEVICE)
        self.this_user_vec = Embedding(1, USER_EMB_SIZE).to(DEVICE)
        max_parameters = [param.numel() for param in self.base_deepsic.parameters()]
        self.hypernetwork = Hypernetwork(USER_EMB_SIZE, max_parameters).to(DEVICE)
        self.hyper_deepsic = HyperDeepSICDetector([param.size() for param in self.base_deepsic.parameters()])
        self.inference_weights = [None for _ in range(MAX_USERS)]

    def _soft_symbols_from_probs(self, input: torch.Tensor, user: int, i: int, hs: List[float]) -> torch.Tensor:
        if i == 1:
            context_embedding = self._get_context_embedding(hs, user)
            self.inference_weights[user] = self.hypernetwork(context_embedding)
        # Set the generated weights to the base network in the forward pass of deepsic
        hyper_input = input.float()
        padding = torch.zeros([hyper_input.shape[0], MAX_USERS - hs.shape[0]]).to(DEVICE)
        hyper_input = torch.cat([hyper_input, padding], dim=1)
        deepsic_output = self.hyper_deepsic(hyper_input, self.inference_weights[user])
        return self.softmax(deepsic_output)

    def _get_context_embedding(self, H: torch.Tensor, user: int) -> torch.Tensor:
        user_embeddings = H
        ind = torch.LongTensor([0]).to(DEVICE)
        context_embedding = []
        for j in range(MAX_USERS):
            if j in range(H.shape[0]):
                if j != user:
                    context_embedding.append(user_embeddings[j].reshape(1, -1))
                else:
                    context_embedding.append(self.this_user_vec(ind))
            else:
                context_embedding.append(self.no_user_vec(ind))
        context_embedding = torch.cat(context_embedding, dim=1)
        return context_embedding

    def train(self, message_words: torch.Tensor, received_words: torch.Tensor, hs: List[List[float]]):
        self.criterion = torch.nn.CrossEntropyLoss()
        total_parameters = self.hypernetwork.parameters()
        total_parameters = chain(total_parameters, self.this_user_vec.parameters())
        total_parameters = chain(total_parameters, self.no_user_vec.parameters())
        self.optimizer = torch.optim.Adam(total_parameters, lr=self.lr)
        for epoch in range(EPOCHS):
            print(f'Epoch {epoch}/{EPOCHS}')
            curr_batch = np.random.choice(len(hs), TRAINING_SYMBOLS)
            for i in curr_batch:
                n_users = hs[i].shape[0]
                for user in range(n_users):
                    mx, rx = message_words[i], received_words[i]
                    h = hs[i]
                    # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                    probs_vec = torch.rand(mx.shape).to(DEVICE)
                    mx_all, rx_all = self._prepare_data_for_training(mx, rx, probs_vec)
                    # get the context embedding for the hypernetwork based on the user and snrs
                    context_embedding = self._get_context_embedding(h, user)
                    # Forward pass through the hypernetwork to generate weights
                    weights = self.hypernetwork(context_embedding)
                    # Set the generated weights to the base network in the forward pass of deepsic
                    hyper_input = rx_all[user].float()
                    padding = torch.zeros([hyper_input.shape[0], MAX_USERS - h.shape[0]]).to(DEVICE)
                    hyper_input = torch.cat([hyper_input, padding], dim=1)
                    soft_estimation = self.hyper_deepsic(hyper_input, weights)
                    # calculate loss
                    loss = self._calc_loss(est=soft_estimation, mx=mx[:, user])
                    # back propagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

    def count_parameters(self):
        params_high = count_parameters(self.hypernetwork)
        return 0, params_high
