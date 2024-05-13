from typing import List

import torch
from torch import nn

from python_code import conf

USER_EMB_SIZE = 16


class UserEmbedder(nn.Module):
    def __init__(self):
        super(UserEmbedder, self).__init__()
        self.fc_embedding = nn.Linear(conf.n_ant + 1, USER_EMB_SIZE)
        self.fc_embedding2 = nn.Linear(USER_EMB_SIZE, USER_EMB_SIZE)
        self.activation = nn.ReLU()

    def forward(self, snr_values: torch.Tensor) -> List[torch.Tensor]:
        embedding = self.fc_embedding(snr_values)
        output = self.fc_embedding2(self.activation(embedding))
        # desired_min = -1.0
        # desired_max = 1.0
        # min_val = torch.min(output)
        # max_val = torch.max(output)
        # normalized_output = desired_min + (output - min_val) * (desired_max - desired_min) / (max_val - min_val)
        return output
