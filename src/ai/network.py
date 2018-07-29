import torch
import torch.nn.functional as F
import torch.nn as nn

class Network(torch.nn.Module):
    def __init__(self, board_size):
        super(Network, self).__init__()

        self.network = nn.Sequential(*[
            torch.nn.Linear(board_size ** 2, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, board_size ** 2)
        ])

    def forward(self, x):
        return self.network(x)

class Loss(nn.Module):
    def __init__(self, board_size):
        super(Loss, self).__init__()

        self.network = nn.Sequential(*[
            torch.nn.LogSoftmax(board_size ** 2)
        ])

    def forward(self, x):
        return self.network(x)