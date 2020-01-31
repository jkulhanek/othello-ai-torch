import torch
import torch.nn.functional as F
import torch.nn as nn
import random

class Network(torch.nn.Module):
    def __init__(self, board_size, initialize = True):
        super(Network, self).__init__()
        self.board_size = board_size

        
        self.network = nn.Sequential(*[
            torch.nn.Linear(board_size ** 2, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, board_size ** 2)
        ])

    def forward(self, x):
        return self.network(x)

def evaluate_network(net, input, actions, device = None):
    board_size = net.board_size
    moves = actions
    vals = net(input.view(-1, board_size ** 2))
    target_multiplier = torch.Tensor(vals.size()).fill_(0.0)

    for (x,y) in moves:
        target_multiplier[0, y * board_size + x] = 1

    move_num = (vals * target_multiplier).max(1)[1].view(1, 1)

    move = torch.cat([move_num % board_size, move_num / board_size], 1)
    return move

def create_player(network):
    def play(board, moves):
        board_size = board.width
        board = torch.tensor(board, dtype=torch.float)        
        vals = network(board.view(-1, board_size ** 2))

        target_multiplier = torch.Tensor(vals.size()).fill_(0.0)

        for (x,y) in moves:
            target_multiplier[0, y * board_size + x] = 1

        move_num = (vals * target_multiplier).max(1)[1].view(1, 1)

        move = torch.cat([move_num % board_size, move_num / board_size], 1)
        print(move)
        return move[0]
    return play

def random_move(board_size):
    n = random.randrange(board_size ** 2)
    return (n // board_size, n % board_size)

class Loss(nn.Module):
    def __init__(self, board_size):
        super(Loss, self).__init__()

        self.network = nn.Sequential(*[
            torch.nn.LogSoftmax(board_size ** 2)
        ])

    def forward(self, x):
        torch.nn.NLLLoss()
        return self.network(x)