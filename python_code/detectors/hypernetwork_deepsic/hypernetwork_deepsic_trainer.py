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
from python_code.utils.constants import TRAINING_TYPES_DICT, TrainingType, HIDDEN_SIZES_DICT, MAX_USERS

EPOCHS = 10


class HypernetworkDeepSICTrainer(DeepSICTrainer):

    def __init__(self):
        super().__init__()
        self.train_context_embedding = []
        self.test_context_embedding = []
        if TRAINING_TYPES_DICT[conf.training_type] == TrainingType.Online:
            raise ValueError("Online training is not implemented for this _detector!!!")

    def __str__(self):
        return TRAINING_TYPES_DICT[conf.training_type].name + ' Hypernetwork-based DeepSIC'

    def _initialize_detector(self):
        self.user_embedder = UserEmbedder().to(DEVICE)
        self.hidden_size = HIDDEN_SIZES_DICT[TrainingType.Online]
        self.base_deepsic = DeepSICDetector(self.hidden_size)
        self.no_user_vec = torch.nn.Parameter(torch.ones([1, USER_EMB_SIZE])).to(DEVICE)
        self.this_user_vec = torch.nn.Parameter(torch.ones([1, USER_EMB_SIZE])).to(DEVICE)
        total_parameters = [param.numel() for param in self.base_deepsic.parameters()]
        self.hypernetwork = Hypernetwork(USER_EMB_SIZE, total_parameters).to(DEVICE)
        self.hyper_deepsic = HyperDeepSICDetector([param.size() for param in self.base_deepsic.parameters()])

    def _soft_symbols_from_probs(self, input: torch.Tensor, user: int, i: int, hs: List[float]) -> torch.Tensor:
        if i == 1:
            context_embedding = self._get_context_embedding(hs, user)
            # self.test_context_embedding.append(context_embedding.cpu().numpy())
            self.inference_weights = [None for _ in range(conf.n_user)]
            self.inference_weights[user] = self.hypernetwork(context_embedding)
        deepsic_output = self.hyper_deepsic(input.float(), self.inference_weights[user])
        return self.softmax(deepsic_output)

    def _get_context_embedding(self, H: torch.Tensor, user: int) -> torch.Tensor:
        user_embeddings = self.user_embedder(torch.Tensor(H).to(DEVICE))
        no_user_vector = -5 * torch.ones([1, USER_EMB_SIZE]).to(DEVICE)
        that_user_vector = 5 * torch.ones([1, USER_EMB_SIZE]).to(DEVICE)
        context_embedding = []
        for j in range(MAX_USERS):
            if j in range(conf.n_user):
                if user != j:
                    context_embedding.append(user_embeddings[j].reshape(1, -1))
                else:
                    context_embedding.append(that_user_vector)
            else:
                context_embedding.append(no_user_vector)
        context_embedding = torch.cat(context_embedding, dim=1)
        return context_embedding

    def train(self, message_words: torch.Tensor, received_words: torch.Tensor, hs: List[List[float]]):
        self.criterion = torch.nn.CrossEntropyLoss()
        total_parameters = self.user_embedder.parameters()
        total_parameters = chain(total_parameters, self.hypernetwork.parameters())
        total_parameters = chain(total_parameters, self.this_user_value)
        total_parameters = chain(total_parameters, self.no_user_value)
        self.optimizer = torch.optim.Adam(total_parameters, lr=self.lr)
        for epoch in range(EPOCHS):
            curr_batch = np.random.choice(len(hs), 50)
            for i in curr_batch:
                for user in range(conf.n_user):
                    mx, rx, h = message_words[i], received_words[i], hs[i]
                    # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                    probs_vec = torch.rand(mx.shape).to(DEVICE)
                    mx_all, rx_all = self._prepare_data_for_training(mx, rx, probs_vec)
                    # get the context embedding for the hypernetwork based on the user and snrs
                    context_embedding = self._get_context_embedding(h, user)
                    # Forward pass through the hypernetwork to generate weights
                    weights = self.hypernetwork(context_embedding)
                    # Set the generated weights to the base network in the forward pass of deepsic
                    soft_estimation = self.hyper_deepsic(rx_all[user].float(), weights)
                    # calculate loss
                    loss = self._calc_loss(est=soft_estimation, mx=mx[:, user])
                    # back propagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        # with torch.no_grad():
        #     for i in range(len(snrs_list)):
        #         for user in range(conf.n_user):
        #             mx, rx, snrs = message_words[i], received_words[i], snrs_list[i]
        #             # get the context embedding for the hypernetwork based on the user and snrs
        #             context_embedding = self._get_context_embedding(snrs, user)
        #             self.train_context_embedding.append(context_embedding.detach().cpu().numpy())
