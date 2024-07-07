import random

import numpy as np
import torch
from torch import nn

from python_code import conf
from python_code.utils.constants import TRAINING_TYPES_DICT, DetectorUtil

random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)


class Detector(nn.Module):
    """
    Sets the foundation for the general symbols detector
    """

    def __init__(self):
        super().__init__()
        self._initialize_detector()
        self.softmax = torch.nn.Softmax(dim=1)
        self.training_type = TRAINING_TYPES_DICT[conf.training_type]

    def get_name(self):
        return self.__name__()

    def _initialize_detector(self):
        """
        Every evaluater must have some base _detector
        """
        self._detector = None

    # calculate train loss
    def _calc_loss(self, est: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
        pass

    def _run_train_loop(self, est: torch.Tensor, mx: torch.Tensor) -> float:
        # calculate loss
        loss = self._calc_loss(est=est, mx=mx)
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss

    def train(self, mx: torch.Tensor, rx: torch.Tensor, detector_util: DetectorUtil):
        pass

    def forward(self, rx: torch.Tensor, detector_util: DetectorUtil) -> torch.Tensor:
        pass

    def count_parameters(self):
        pass
