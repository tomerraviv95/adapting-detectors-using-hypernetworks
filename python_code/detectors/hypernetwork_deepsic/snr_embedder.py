from typing import List

import torch
from torch import nn


class UserEmbedder(nn.Module):
    def __init__(self, emb_size: int):
        super(UserEmbedder, self).__init__()
        self.activation = nn.ReLU()
        self.fc_embedding = nn.Linear(1, emb_size)
        self.fc_embedding2 = nn.Linear(emb_size, emb_size)

    def forward(self, snr_values: torch.Tensor) -> List[torch.Tensor]:
        mid = self.activation(self.fc_embedding(snr_values))
        embedding = self.activation(self.fc_embedding2(mid))
        return embedding
