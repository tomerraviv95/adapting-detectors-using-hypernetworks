from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.detectors.deepsic_detector import DeepSICDetector
from python_code.detectors.deepsic_trainer import DeepSICTrainer, EPOCHS


class RecDeepSICTrainer(DeepSICTrainer):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'F-DeepSIC'

    def _initialize_detector(self):
        # populate 2D list for Storing the DeepSIC Networks
        self.detector = []
        for _ in range(conf.n_user):
            deepsic = DeepSICDetector()
            self.detector.append([deepsic for _ in range(self.iterations)])

    def train_model(self, single_model: nn.Module, mx: torch.Tensor, rx: torch.Tensor):
        """
        Trains a DeepSIC Network
        """
        self.optimizer = torch.optim.Adam(single_model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        single_model = single_model.to(DEVICE)
        y_total = rx.float()
        for _ in range(EPOCHS):
            soft_estimation = single_model(y_total)
            self.run_train_loop(soft_estimation, mx)

    def train_models(self, model: List[DeepSICDetector], i: int, tx_all: List[torch.Tensor],
                     rx_all: List[torch.Tensor]):
        for user in range(conf.n_user):
            self.train_model(model[user][i], tx_all[user], rx_all[user])
