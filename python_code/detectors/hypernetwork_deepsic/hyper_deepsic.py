from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from python_code.utils.config_singleton import Config

conf = Config()


class HyperDeepSICDetector(nn.Module):
    """
    The hypernetwork version of DeepSIC

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
        self.activation = nn.ReLU()

    def forward(self, rx: torch.Tensor, var: List[torch.Tensor]) -> torch.Tensor:
        mid = self.activation(F.linear(rx, weight=var[0].reshape(self.sizes[0]), bias=var[1].reshape(self.sizes[1])))
        out = F.linear(mid, weight=var[2].reshape(self.sizes[2]), bias=var[3].reshape(self.sizes[3]))
        return out
