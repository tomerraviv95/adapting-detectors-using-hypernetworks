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
        self.embedding2 = nn.Linear(EMB, EMB//2)
        self.embedding3 = nn.Linear(EMB//2, EMB//4)
        self.fc_outs = nn.ModuleList(
            [nn.Linear(EMB//4, cur_params) for cur_params in parameters_num])

    def forward(self, rx: torch.Tensor) -> List[torch.Tensor]:
        embedding1 = self.activation(self.embedding(rx))
        embedding2 = self.activation(self.embedding2(embedding1))
        embedding3 = self.activation(self.embedding3(embedding2))
        fc_weights = []
        for i in range(len(self.fc_outs)):
            fc_weight = self.fc_outs[i](embedding3)
            fc_weights.append(fc_weight)
        return fc_weights
