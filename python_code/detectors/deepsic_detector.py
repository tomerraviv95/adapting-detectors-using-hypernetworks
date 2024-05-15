import torch
from torch import nn

from python_code import conf


class DeepSICDetector(nn.Module):
    """
    The DeepSIC Network Architecture

    ===========Architecture=========
    DeepSICNet(
      (fullyConnectedLayer): Linear(...)
      (reluLayer): ReLU()
      (fullyConnectedLayer2): Linear(...)
    ================================
    """

    def __init__(self, n_user: int, hidden_size: int):
        super(DeepSICDetector, self).__init__()
        linear_input = conf.n_ant + (n_user - 1)  # from DeepSIC paper
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        mid = self.activation(self.fc1(rx))
        out = self.fc2(mid)
        return out
