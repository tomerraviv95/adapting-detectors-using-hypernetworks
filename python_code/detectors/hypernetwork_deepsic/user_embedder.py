from typing import List

import torch
from torch import nn

from python_code import conf

USER_EMB_SIZE = 16


class UserEmbedder(nn.Module):
    def __init__(self):
        super(UserEmbedder, self).__init__()
        self.fc_embedding = nn.Linear(conf.n_ant, USER_EMB_SIZE)
        self.activation = nn.ReLU()

    def forward(self, snr_values: torch.Tensor) -> List[torch.Tensor]:
        embedding = self.fc_embedding(snr_values)
        return self.activation(embedding)
