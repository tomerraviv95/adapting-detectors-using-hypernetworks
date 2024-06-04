from typing import List

import torch
from torch import nn

from python_code.utils.constants import MAX_USERS

EMB = 64


class Hypernetwork(nn.Module):
    def __init__(self, input_size: int, parameters_num: List[int]):
        super(Hypernetwork, self).__init__()
        self.activation = nn.ReLU()
        self.embedding = nn.Linear(MAX_USERS * input_size, EMB)
        self.embedding2 = nn.Linear(EMB, EMB // 2)
        self.fc_output = nn.Linear(EMB // 2, sum(parameters_num))
        self.parameters_num = parameters_num

    def forward(self, rx: torch.Tensor) -> List[torch.Tensor]:
        embedding1 = self.activation(self.embedding(rx))
        embedding2 = self.activation(self.embedding2(embedding1))
        out = self.fc_output(embedding2)
        fc_weights = []
        start = 0
        for param_num in self.parameters_num:
            fc_weights.append(out[:, start:start + param_num])
            start += param_num
        return fc_weights
