import game.board_utils as butils
import random
import torch
import copy

class Environment:
    def __init__(self, board_size):
        super(Environment, self).__init__()
        self.board_size = board_size
        self.opponent = lambda board, actions: random.choice(actions)
        self.reset()

    def reset(self):
        self.board = butils.Board(iterable = [[butils.EMPTY] * self.board_size for i in range(self.board_size)])
        startx = self.board_size //2 - 1
        self.board[startx][startx] = butils.PLAYER_1
        self.board[startx][startx + 1] = butils.PLAYER_2
        self.board[startx + 1][startx] = butils.PLAYER_2
        self.board[startx + 1][startx + 1] = butils.PLAYER_1

    def set_opponent(self, opponent):
        self.opponent = opponent

    def get_board(self, device = None):
        return torch.tensor(self.board, dtype = torch.float, device = device)

    def is_end(self):
        bd_flatten = [x for x in set().union(*self.board)]
        if len([x for x in bd_flatten if x == butils.PLAYER_1]) == 0:
            return True
        if len([x for x in bd_flatten if x == butils.PLAYER_2]) == 0:
            return True

        #test for player's options
        bs=butils.Board(iterable = self.board)
        potentials = butils.get_potentials(bs, butils.PLAYER_1)
        potentials.extend(butils.get_potentials(bs,  butils.PLAYER_2))

        if len(potentials) == 0:
            return True
        return False

    def actions(self):
        return butils.get_potentials(self.board, butils.PLAYER_1)

    def step(self, action):
        self.board = butils.apply_move(self.board, butils.PLAYER_1, action)
        while not self.is_end():
            opponents_actions = butils.get_potentials(self.board, butils.PLAYER_2)
            if len(opponents_actions) == 0:
                break
            
            opponents_move = self.opponent(self.board, opponents_actions)
            self.board = butils.apply_move(self.board, butils.PLAYER_2, opponents_move)

            if len(butils.get_potentials(self.board, butils.PLAYER_1)) > 0:
                break
        return None, float(0 if not self.is_end() else butils.get_final_result(self.board, butils.PLAYER_1)), self.is_end(), None

    def get_score(self):
        return butils.get_final_result(self.board, butils.PLAYER_1)