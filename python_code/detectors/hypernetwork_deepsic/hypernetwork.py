from typing import List

import torch
from torch import nn

from python_code import conf

EMB = 256

class Hypernetwork(nn.Module):
    def __init__(self, input_size: int, parameters_num: List[int]):
        super(Hypernetwork, self).__init__()
        self.activation = nn.ReLU()
        self.embedding = nn.Linear((conf.n_user - 1) * input_size, EMB)
        self.fc_outs = nn.ModuleList(
            [nn.Linear(EMB, cur_params) for cur_params in parameters_num])

    def forward(self, rx: torch.Tensor) -> List[torch.Tensor]:
        embedding = self.activation(self.embedding(rx))
        fc_weights = []
        for i in range(len(self.fc_outs)):
            fc_weight = self.fc_outs[i](embedding)
            fc_weights.append(fc_weight)
        return fc_weights
