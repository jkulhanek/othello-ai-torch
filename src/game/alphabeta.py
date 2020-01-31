import game.board_utils as butils
import copy


_ab_simple_weights = [[120.0/1176.0,-20.0/1176.0,20.0/1176.0,5.0/1176.0,5.0/1176.0,20.0/1176.0,-20.0/1176.0,120.0/1176.0],
    [-20.0/1176.0,-40.0/1176.0,-5.0/1176.0,-5.0/1176.0,-5.0/1176.0,-5.0/1176.0,-40.0/1176.0,-20.0/1176.0],
    [20.0/1176.0,-5.0/1176.0,15.0/1176.0,3.0/1176.0,3.0/1176.0,15.0/1176.0,-5.0/1176.0,20.0/1176.0],
    [5.0/1176.0,-5.0/1176.0,3.0/1176.0,3.0/1176.0,3.0/1176.0,3.0/1176.0,-5.0/1176.0,5.0/1176.0],
    [5.0/1176.0,-5.0/1176.0,3.0/1176.0,3.0/1176.0,3.0/1176.0,3.0/1176.0,-5.0/1176.0,5.0/1176.0],
    [20.0/1176.0,-5.0/1176.0,15.0/1176.0,3.0/1176.0,3.0/1176.0,15.0/1176.0,-5.0/1176.0,20.0/1176.0],
    [-20.0/1176.0,-40.0/1176.0,-5.0/1176.0,-5.0/1176.0,-5.0/1176.0,-5.0/1176.0,-40.0/1176.0,-20.0/1176.0],
    [120.0/1176.0,-20.0/1176.0,20.0/1176.0,5.0/1176.0,5.0/1176.0,20.0/1176.0,-20.0/1176.0,120.0/1176.0]]

def _ab_simple_ranking_function(board):
    '''default function if no other is presented, only for 8x8'''
    players_dict={
        butils.EMPTY:0,
        butils.PLAYER_1:1,
        butils.PLAYER_2:-1
    }
    tpl=0
    for (y,row) in enumerate(board):
        tpl+=sum([coef*players_dict[board_data] for (board_data, coef) in zip(row,_ab_simple_weights[y])])
    return tpl  

def create_alphabeta_player(max_depth = 6, ranking_function = None, max_value = None):
    if ranking_function:
        max_value = max_value if max_value else 1
        ranking_function = ranking_function
    else:
        max_value =1.0
        ranking_function = _ab_simple_ranking_function

    def alpha_beta(board, player, depth=1, alpha=None, beta=None,cancellation_token=None):
        if alpha is None:
            alpha=-max_value
        if beta is None:
            beta=max_value

        if depth==max_depth or (not cancellation_token is None and cancellation_token.is_cancelled()):
            return (ranking_function(board),None)
        
        potentials=butils.get_potentials(board, player)
        if not potentials:
            #passes game to other player
            #or ends the game
            res=butils.get_final_result(board, player)
            if res is not False:
                sign=1 if res==player else -1
                return (max_value*sign,None)
            else:
                #passes to the other player
                value=ranking_function(player)
                return (value, None)

        best_move=None
        for potential in potentials:
            #emulate the potential
            board_s=butils.apply_move(board, player, potential)
            value=-alpha_beta(board_s, butils.get_opponent(player),depth+1, -beta, -alpha, cancellation_token)[0]
            if value>alpha:
                alpha=value
                best_move=potential
            if beta<=alpha:
                break

        return (alpha, best_move)

    def alphabeta_player(board, moves):
        res=alpha_beta(board, butils.PLAYER_1)
        return res[1]

    return alphabeta_player
        
        

   