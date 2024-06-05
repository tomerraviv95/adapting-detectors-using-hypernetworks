import torch
from torch import nn

from python_code import conf
from python_code.utils.constants import USER_EMB_SIZE


class UserEmbedder(nn.Module):
    def __init__(self):
        super(UserEmbedder, self).__init__()
        self.fc_embedding = nn.Linear(conf.n_ant, USER_EMB_SIZE)

    def forward(self, snr_values: torch.Tensor) -> torch.Tensor:
        embedding = self.fc_embedding(snr_values)
        return embedding