import gui.random_player as random_player
from gui.game_board import GameBoard
from gui.reversi_view import ReversiView
import ai.network
import time
import multiprocessing 
import copy, getopt, sys
import game.board_utils as butils
import game.player_adapter
import game.alphabeta
from gui.reversi_creator import ReversiCreator
import torch

BOARD_SIZE = 8

if __name__ == "__main__": 
    network = ai.network.Network(BOARD_SIZE)
    network.load_state_dict(torch.load("model.pth"))
    network.eval()
    players_dict = {
        
        'random':random_player.MyPlayer,
         'ab': game.player_adapter.create_player(game.alphabeta.create_alphabeta_player(), name = "Alpha beta"),
         'ai': game.player_adapter.create_player(ai.network.create_player(network))}
    game = ReversiCreator(players_dict)
    game.gui.root.mainloop()