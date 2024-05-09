from typing import List

import torch
from torch import nn

USER_EMB_SIZE = 8


class UserEmbedder(nn.Module):
    def __init__(self):
        super(UserEmbedder, self).__init__()
        self.activation = nn.ReLU()
        self.fc_embedding = nn.Linear(1, USER_EMB_SIZE)

    def forward(self, snr_values: torch.Tensor) -> List[torch.Tensor]:
        embedding = self.activation(self.fc_embedding(snr_values))
        return embedding
