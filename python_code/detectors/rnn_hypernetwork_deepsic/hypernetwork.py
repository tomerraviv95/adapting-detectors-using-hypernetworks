from typing import List

import torch
from torch import nn
from torch.nn import RNN

from python_code import DEVICE
from python_code.utils.constants import MAX_USERS

EMB = 24
NUM_LAYERS = 3


class Hypernetwork(nn.Module):
    def __init__(self, input_size: int, parameters_num: List[int]):
        super(Hypernetwork, self).__init__()
        self.rnn = RNN(input_size=MAX_USERS * input_size, hidden_size=EMB, num_layers=NUM_LAYERS, nonlinearity='relu')
        self.fc_outs = nn.ModuleList(
            [nn.Linear(EMB, cur_params) for cur_params in parameters_num])
        self.hidden_states = {k: torch.zeros([NUM_LAYERS, EMB]).to(DEVICE) for k in range(MAX_USERS)}

    def reset_hidden_state(self, user: int):
        self.hidden_states[user] = torch.zeros([NUM_LAYERS, EMB]).to(DEVICE)

    def reset_hidden_states(self, prev_users, cur_user):
        for user in range(prev_users, cur_user):
            self.reset_hidden_state(user)

    def forward(self, rx: torch.Tensor, user: int) -> List[torch.Tensor]:
        embedding, hidden = self.rnn(rx, self.hidden_states[user])
        self.hidden_states[user] = hidden.detach()
        fc_weights = []
        for i in range(len(self.fc_outs)):
            fc_weight = self.fc_outs[i](embedding)
            fc_weights.append(fc_weight)
        return fc_weights
