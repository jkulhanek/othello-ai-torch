import gui.random_player as random_player
from gui.game_board import GameBoard
from gui.reversi_view import ReversiView
import time
import multiprocessing 
import copy, getopt, sys
import game.board_utils as butils
import game.player_adapter
import game.alphabeta
from gui.reversi_creator import ReversiCreator

if __name__ == "__main__": 
    players_dict = {'random':random_player.MyPlayer, 'ab': game.player_adapter.create_player(game.alphabeta.create_alphabeta_player(), name = "Alpha beta")}
    game = ReversiCreator(players_dict)
    game.gui.root.mainloop()