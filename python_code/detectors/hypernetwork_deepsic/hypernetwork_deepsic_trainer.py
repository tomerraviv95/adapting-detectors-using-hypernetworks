from itertools import chain

import numpy as np
import torch
from torch.nn import Embedding

from python_code import DEVICE
from python_code.detectors.deepsic.deepsic_detector import DeepSICDetector
from python_code.detectors.hypernetwork_deepsic.hyper_deepsic import HyperDeepSICDetector
from python_code.detectors.hypernetwork_deepsic.hypernetwork import Hypernetwork
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import MAX_USERS, HIDDEN_SIZE, DetectorUtil, TRAINING_BLOCKS_PER_CONFIG
from python_code.utils.metrics import count_parameters

BATCH_SIZE = 32


class HypernetworkTrainer(Trainer):

    def __init__(self):
        super().__init__()
        self.lr = 5e-4
        self.epochs = 30
        self.train_context_embedding = []
        self.test_context_embedding = []

    def __str__(self):
        return 'Hypernetwork-based DeepSIC'

    def _initialize_detector(self):
        self.hidden_size = HIDDEN_SIZE
        self.base_deepsic = DeepSICDetector(MAX_USERS, self.hidden_size)
        self.no_user_vec = Embedding(1, MAX_USERS).to(DEVICE)
        self.this_user_vec = Embedding(1, MAX_USERS).to(DEVICE)
        max_parameters = [param.numel() for param in self.base_deepsic.parameters()]
        self.hypernetwork = Hypernetwork(MAX_USERS, max_parameters).to(DEVICE)
        self.hyper_deepsic = HyperDeepSICDetector([param.size() for param in self.base_deepsic.parameters()])
        self.inference_weights = [None for _ in range(MAX_USERS)]

    def _soft_symbols_from_probs(self, input: torch.Tensor, user: int, detector_util: DetectorUtil,
                                 i: int) -> torch.Tensor:
        if i == 1:
            # set the generated weights to the base network in the first iteration
            context_embedding = self._get_context_embedding(detector_util.H_hat, user)
            self.inference_weights[user] = self.hypernetwork(context_embedding)
        hyper_input = input.float()
        padding = torch.zeros([hyper_input.shape[0], MAX_USERS - detector_util.H_hat.shape[0]]).to(DEVICE)
        hyper_input = torch.cat([hyper_input, padding], dim=1)
        deepsic_output = self.hyper_deepsic(hyper_input, self.inference_weights[user])
        return self.softmax(deepsic_output)

    def _get_context_embedding(self, h: torch.Tensor, user: int) -> torch.Tensor:
        ind = torch.LongTensor([0]).to(DEVICE)
        context_embedding = []
        for j in range(MAX_USERS):
            if j in range(h.shape[0]):
                if j != user:
                    context_embedding.append(h[j].reshape(1, -1))
                else:
                    context_embedding.append(self.this_user_vec(ind))
            else:
                context_embedding.append(self.no_user_vec(ind))
        context_embedding = torch.cat(context_embedding, dim=1)
        return context_embedding

    def train(self, message_words: torch.Tensor, received_words: torch.Tensor, detector_util: DetectorUtil):
        H_hat = detector_util.H_hat
        self.criterion = torch.nn.CrossEntropyLoss()
        total_parameters = self.hypernetwork.parameters()
        total_parameters = chain(total_parameters, self.this_user_vec.parameters())
        total_parameters = chain(total_parameters, self.no_user_vec.parameters())
        self.optimizer = torch.optim.Adam(total_parameters, lr=self.lr)
        all_users_indices = TRAINING_BLOCKS_PER_CONFIG * np.arange(0, len(message_words) // TRAINING_BLOCKS_PER_CONFIG)
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            # to make sure that we sample over all possible user values
            # and minimize their losses simultaneously - this is the heart of the method!
            curr_batch = np.random.choice(TRAINING_BLOCKS_PER_CONFIG, BATCH_SIZE).reshape(-1, 1) + all_users_indices.reshape(1,-1)
            curr_batch = curr_batch.reshape(-1)
            total_loss = 0
            for i in curr_batch:
                loss = 0
                n_users = H_hat[i].shape[0]
                for user in range(n_users):
                    mx, rx = message_words[i], received_words[i]
                    h = H_hat[i]
                    # obtaining the DeepSIC networks for each user-symbol and the i-th iteration
                    probs_vec = torch.rand(mx.shape).to(DEVICE)
                    mx_all, rx_all = self._prepare_data_for_training(mx, rx, probs_vec)
                    # get the context embedding for the hypernetwork based on the user and snrs
                    context_embedding = self._get_context_embedding(h, user)
                    # forward pass through the hypernetwork to generate weights
                    weights = self.hypernetwork(context_embedding)
                    # set the generated weights to the base network in the forward pass of deepsic
                    hyper_input = rx_all[user].float()
                    padding = torch.zeros([hyper_input.shape[0], MAX_USERS - h.shape[0]]).to(DEVICE)
                    hyper_input = torch.cat([hyper_input, padding], dim=1)
                    soft_estimation = self.hyper_deepsic(hyper_input, weights)
                    # calculate loss
                    loss += self._calc_loss(est=soft_estimation, mx=mx[:, user])
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss / n_users
            avg_loss = (total_loss / len(curr_batch)).item()
            print(f"Loss: {avg_loss}")

    def count_parameters(self):
        total_params = count_parameters(self.hypernetwork)
        total_params += count_parameters(self.no_user_vec)
        total_params += count_parameters(self.this_user_vec)
        print(f"Hypernetwork + embeddings params: {total_params}")
