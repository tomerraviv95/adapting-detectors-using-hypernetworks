from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from python_code.utils.config_singleton import Config

conf = Config()


class HyperDeepSICDetector(nn.Module):
    """
    The Hyper DeepSIC Network Architecture

    ===========Architecture=========
    DeepSICNet(
      (fullyConnectedLayer): Linear(...)
      (reluLayer): ReLU()
      (fullyConnectedLayer2): Linear(...)
    ================================
    """

    def __init__(self, sizes: List[int]):
        super(HyperDeepSICDetector, self).__init__()
        self.sizes = sizes

    def forward(self, rx: torch.Tensor, var: List[torch.Tensor]) -> torch.Tensor:
        mid = torch.relu(F.linear(rx, var[0].reshape(self.sizes[0]), var[1].reshape(self.sizes[1])))
        out = F.linear(mid, var[2].reshape(self.sizes[2]), var[3].reshape(self.sizes[3]))
        return out
