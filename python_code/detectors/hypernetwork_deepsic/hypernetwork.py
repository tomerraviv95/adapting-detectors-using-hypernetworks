from torch import nn


class Hypernetwork(nn.Module):
    def __init__(self):
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, classes_num)
