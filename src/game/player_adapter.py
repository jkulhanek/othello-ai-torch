import game.board_utils as butils


def create_player(func, name = "Player"):
    class _Player(object):
        def __init__(self, my_color, opponent_color):
            self.name = name 
            self.my_color = my_color
            self.opponent_color = opponent_color
            #score in tournament
            self.total_score=0
 
        def move(self, board):
            board_normal = butils.normalize(board, self.my_color)
            board_set = butils.Board(iterable = board_normal)
            potentials = butils.get_potentials(board_set, self.my_color) #butils.PLAYER_1)

            if len(potentials) == 0:
                return False
            
            return func(board = board_set, moves = potentials, color = butils.PLAYER_1) self.my_color)          
    return _Player