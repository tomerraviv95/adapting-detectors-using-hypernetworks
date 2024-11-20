import torch
from torch import nn

from python_code import DEVICE
from python_code.datasets.communications_blocks.modulator import BPSKModulator
from python_code.utils.constants import HIDDEN_SIZE, HALF, DetectorUtil
from python_code.utils.metrics import count_parameters


class Trainer(nn.Module):

    def __init__(self):
        self.iterations = 3
        self.hidden_size = HIDDEN_SIZE
        super().__init__()
        self._initialize_detector()
        self.softmax = torch.nn.Softmax(dim=1)

    def __str__(self):
        return 'DeepSIC'

    def _initialize_detector(self):
        pass

    def _calc_loss(self, est: torch.Tensor, mx: torch.IntTensor) -> torch.Tensor:
        """
        Cross entropy loss - distribution over states versus the gt state label
        """
        return self.criterion(input=est, target=mx.long())

    def _symbols_from_prob(self, probs_vec: torch.Tensor) -> torch.Tensor:
        symbols_word = torch.sign(probs_vec - HALF)
        detected_words = BPSKModulator.demodulate(symbols_word)
        return detected_words

    def _prepare_data_for_training(self, mx: torch.Tensor, rx: torch.Tensor, probs_vec: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
        """
        Generates the data for each user
        """
        mx_all = []
        rx_all = []
        for k in range(mx.shape[1]):
            idx = [user_i for user_i in range(mx.shape[1]) if user_i != k]
            current_y_train = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            mx_all.append(mx[:, k])
            rx_all.append(current_y_train)
        return mx_all, rx_all

    def _calculate_posteriors(self, i: int, probs_vec: torch.Tensor, rx: torch.Tensor,
                              detector_util: DetectorUtil = None) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks for a single iteration.
        """
        next_probs_vec = torch.zeros(probs_vec.shape).to(DEVICE)
        for user in range(probs_vec.shape[1]):
            idx = [user_i for user_i in range(probs_vec.shape[1]) if user_i != user]
            input = torch.cat((rx, probs_vec[:, idx].reshape(rx.shape[0], -1)), dim=1)
            with torch.no_grad():
                output = self._soft_symbols_from_probs(input, user, detector_util, i)
            next_probs_vec[:, user] = output[:, 1:].reshape(next_probs_vec[:, user].shape)
        return next_probs_vec

    def forward(self, rx: torch.Tensor, detector_util: DetectorUtil) -> torch.Tensor:
        """
        Detects the batch of received words -> estimated transmitted words
        """
        with torch.no_grad():
            self.adapt_network(detector_util.n_users)
            # initialize the states of new users entering the network
            probs_vec = HALF * torch.ones([rx.shape[0], detector_util.n_users]).to(DEVICE).float()
            # detect using all iterations
            for i in range(self.iterations):
                probs_vec = self._calculate_posteriors(i + 1, probs_vec, rx, detector_util)
            # get symbols from probs
            detected_words = self._symbols_from_prob(probs_vec)
            self.prev_users = detector_util.n_users
            return detected_words

    def adapt_network(self,n_users):
        pass

    def count_parameters(self):
        smallest_model = list(self.detector.values())[0][0]
        largest_model = list(self.detector.values())[-1][0]
        params_low = count_parameters(smallest_model)
        params_high = count_parameters(largest_model)
        print(f"Smallest module params: {params_low}, largest module params: {params_high}")

    def _run_train_loop(self, est: torch.Tensor, mx: torch.Tensor) -> float:
        # calculate loss
        loss = self._calc_loss(est=est, mx=mx)
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss

    def train_model(self, single_model: nn.Module, tx: torch.Tensor, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        self.optimizer = torch.optim.Adam(single_model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        single_model = single_model.to(DEVICE)
        loss = 0
        for _ in range(self.epochs):
            soft_estimation = single_model(rx.float())
            current_loss = self._run_train_loop(soft_estimation, tx)
            loss += current_loss
