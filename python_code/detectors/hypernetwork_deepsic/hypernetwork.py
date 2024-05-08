from typing import List

import torch
from torch import nn

from python_code import conf


class Hypernetwork(nn.Module):
    def __init__(self, parameters_num: List[int]):
        super(Hypernetwork, self).__init__()
        classes_num = 2
        linear_input = conf.n_ant + (classes_num - 1) * (conf.n_user - 1)
        emb_size = 24
        self.activation = nn.ReLU()
        self.fc_embedding = nn.Linear(linear_input, emb_size)
        self.fc_outs = nn.ModuleList([nn.Linear(emb_size, cur_params) for cur_params in parameters_num])

    def forward(self, rx: torch.Tensor) -> List[torch.Tensor]:
        embedding = self.activation(self.fc_embedding(rx))
        fc_weights = []
        for i in range(len(self.fc_outs)):
            fc_weight = self.fc_outs[i](embedding)
            fc_weights.append(fc_weight)
        return fc_weights
