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
from python_code.utils.constants import TRAINING_TYPES_DICT, TrainingType, HIDDEN_SIZES_DICT, EPOCHS_DICT

EPOCHS = 50


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
        self.base_deepsics = [DeepSICDetector(self.hidden_size).to(DEVICE) for _ in range(conf.n_user)]
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
        context_embedding = 0
        for j in range(conf.n_user):
            if user != j:
                context_embedding += user_embeddings[j].reshape(1, -1)
        return context_embedding

    def train(self, message_words: torch.Tensor, received_words: torch.Tensor, snrs_list: List[List[float]]):
        ####################################################################
        ## initial model training ##
        total_parameters = self.base_deepsics[0].parameters()
        for user in range(1, conf.n_user):
            total_parameters = chain(total_parameters, self.base_deepsics[user].parameters())
        self.optimizer = torch.optim.Adam(total_parameters, lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion2 = torch.nn.MSELoss()
        mx = message_words.reshape(-1, message_words.shape[2])
        rx = received_words.reshape(-1, received_words.shape[2])
        probs_vec = 0.5 * torch.ones(mx.shape).to(DEVICE)
        # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
        mx_all, rx_all = self._prepare_data_for_training(mx, rx, probs_vec)
        epochs = EPOCHS_DICT[conf.training_type]
        self.weights_per_user = [[] for _ in range(conf.n_user)]
        for user in range(conf.n_user):
            single_model = self.base_deepsics[user]
            y_total = rx_all[user].float()
            for _ in range(epochs):
                soft_estimation = single_model(y_total)
                self._run_train_loop(soft_estimation, mx_all[user])
            self.weights_per_user[user].extend(list(single_model.parameters()))
        ####################################################################
        total_parameters = self.user_embedder.parameters()
        for user in range(conf.n_user):
            total_parameters = chain(total_parameters, self.hypernetworks[user].parameters())
        self.optimizer = torch.optim.Adam(total_parameters, lr=self.lr)
        for epoch in range(EPOCHS):
            curr_batch = np.random.choice(len(snrs_list), 20)
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
                    mse_loss = 0
                    for pred_weight, gt_weight in zip(weights, self.weights_per_user[user]):
                        mse_loss += self.criterion2(input=pred_weight.reshape(-1), target=gt_weight.reshape(-1))
                    # Set the generated weights to the base network in the forward pass of deepsic
                    soft_estimation = self.hyper_deepsic(rx_all[user].float(), weights)
                    # calculate loss
                    ce_loss = self._calc_loss(est=soft_estimation, mx=mx[:, user])
                    loss = ce_loss + mse_loss
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
