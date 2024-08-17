from typing import List

import torch
from torch import nn

from python_code.utils.constants import MAX_USERS

EMB = 64


class Hypernetwork(nn.Module):
    """
    The weights generation network, see Figure 2 in the paper
    """

    def __init__(self, input_size: int, parameters_num: List[int]):
        super(Hypernetwork, self).__init__()
        self.activation = nn.ReLU()
        self.embedding = nn.Linear(MAX_USERS * input_size, EMB)
        self.embedding2 = nn.Linear(EMB, EMB // 2)
        self.fc_outs = nn.ModuleList(
            [nn.Linear(EMB // 2, cur_params) for cur_params in parameters_num])

    def forward(self, rx: torch.Tensor) -> List[torch.Tensor]:
        embedding1 = self.activation(self.embedding(rx))
        embedding2 = self.activation(self.embedding2(embedding1))
        fc_weights = []
        for i in range(len(self.fc_outs)):
            fc_weight = self.fc_outs[i](embedding2)
            fc_weights.append(fc_weight)
        return fc_weights
