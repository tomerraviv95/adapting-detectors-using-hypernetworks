from typing import List

import torch
from torch import nn

EMB_SIZE = 16


class Hypernetwork(nn.Module):
    def __init__(self, input_size: int, parameters_num: List[int]):
        super(Hypernetwork, self).__init__()
        self.activation = nn.ReLU()
        self.fc_embedding = nn.Linear(input_size, EMB_SIZE)
        self.fc_embedding2 = nn.Linear(EMB_SIZE, EMB_SIZE)
        self.fc_outs = nn.ModuleList([nn.Linear(EMB_SIZE, cur_params) for cur_params in parameters_num])

    def forward(self, rx: torch.Tensor) -> List[torch.Tensor]:
        mid = self.activation(self.fc_embedding(rx))
        embedding = self.activation(self.fc_embedding2(mid))
        fc_weights = []
        for i in range(len(self.fc_outs)):
            fc_weight = self.fc_outs[i](embedding)
            fc_weights.append(fc_weight)
        return fc_weights
