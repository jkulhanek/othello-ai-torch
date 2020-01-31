from itertools import takewhile
import copy
import time

EMPTY=0
PLAYER_1=1
PLAYER_2=-1
TIE=0
WIN_BONUS=99999999

def get_potentials(board, player):
    '''faster way to find potentials'''
    opponent=PLAYER_1 if player == PLAYER_2 else PLAYER_2
    playerMapping={
        EMPTY: 0,
        player: 1,
        opponent: -1
    }
    def yield_potential():
        pre_potential=None
        unlocked=False
        lastVal=0
        #first line
        for y in range(board.height):
            pre_potential=None
            unlocked=False
            lastVal=0
            for x in range(board.width):
                val=playerMapping[board[x][y]]
                if (lastVal-val)==-2:
                    if pre_potential!=None:
                        yield pre_potential
                    pre_potential=None
                    unlocked=False
                if (lastVal-val)==2:
                    pre_potential=None
                    unlocked=True
                if val==0:
                    pre_potential=(x,y)
                    if unlocked:
                        yield pre_potential
                        pre_potential=None
                        unlocked=False                        
                lastVal=val
        #second line
        for x in range(board.width):  
            pre_potential=None
            unlocked=False
            lastVal=0  
            for y in range(board.height):
                val=playerMapping[board[x][y]]
                if (lastVal-val)==-2:
                    if pre_potential!=None:
                        yield pre_potential
                    pre_potential=None
                    unlocked=False
                if (lastVal-val)==2:
                    pre_potential=None
                    unlocked=True
                if val==0:
                    pre_potential=(x,y)
                    if unlocked:
                        yield pre_potential
                        pre_potential=None
                        unlocked=False
                lastVal=val
        #first diagonal
        #board must be a square
        l=len(board)
        for k in range(2,l*2-3):
            x0=min(k, l-1)
            y0=max(k,l-1)+1-l
            su=l-abs(k-l+1)
            pre_potential=None
            unlocked=False
            lastVal=0 
            for j in range(su):
                x=x0-j
                y=y0+j
                val=playerMapping[board[x][y]]
                if (lastVal-val)==-2:
                    if pre_potential!=None:
                        yield pre_potential
                    pre_potential=None
                    unlocked=False
                if (lastVal-val)==2:
                    pre_potential=None
                    unlocked=True
                if val==0:
                    pre_potential=(x,y)
                    if unlocked:
                        yield pre_potential
                        pre_potential=None
                        unlocked=False
                lastVal=val
        #second diagonal
        for k in range(2,l*2-3):
            y0=max(0, l-k-1)
            x0=max(k,l-1)+1-l
            su=l-abs(k-l+1)
            pre_potential=None
            unlocked=False
            lastVal=0 
            for j in range(su):                    
                x=x0+j
                y=y0+j
                val=playerMapping[board[x][y]]
                if (lastVal-val)==-2:
                    if pre_potential!=None:
                        yield pre_potential
                    pre_potential=None
                    unlocked=False
                if (lastVal-val)==2:
                    pre_potential=None
                    unlocked=True
                if val==0:
                    pre_potential=(x,y)
                    if unlocked:
                        yield pre_potential
                        pre_potential=None
                        unlocked=False
                lastVal=val
    potentials=list(set(yield_potential()))
    return potentials

def get_turned_stones(board,position, player):
    original=board[position[0]][position[1]]
    board[position[0]][position[1]]=player
    opponent=PLAYER_1 if player==PLAYER_2 else PLAYER_2
    
    turn=[]

    width=len(board[0])
    height=len(board)
    #x =>
    f=[(i, val) for (i, val) in enumerate(board[position[0]]) if i>position[1]]
    pot=list(takewhile(lambda c:c[1]==opponent,f))
    if len(pot)>0 and pot[-1][0]<width-1 and board[position[0]][pot[-1][0]+1]==player:
        turn.extend([(position[0],x[0]) for x in pot])

    #x <=
    pot=list(takewhile(lambda c:c[1]==opponent,[(i, val) for (i, val) in reversed(list(enumerate(board[position[0]]))) if i<position[1]]))
    if len(pot)>0 and pot[-1][0]>0 and board[position[0]][pot[-1][0]-1]==player:
        turn.extend([(position[0],x[0]) for x in pot])

    #y =>
    pot=list(takewhile(lambda c:c[1]==opponent,[(i, row[position[1]]) for (i, row) in enumerate(board) if i>position[0]]))
    if len(pot)>0 and (pot[-1][0]<height-1 and board[pot[-1][0]+1][position[1]]==player):
        turn.extend([(x[0],position[1]) for x in pot])

    #y <=
    pot=list(takewhile(lambda c:c[1]==opponent,[(i, row[position[1]]) for (i, row) in reversed(list(enumerate(board))) if i<position[0]]))
    if len(pot)>0 and pot[-1][0]>0 and board[pot[-1][0]-1][position[1]]==player:
        turn.extend([(x[0],position[1]) for x in pot])

    def onBoard(pos):
        return pos[0]>=0 and pos[1]>=0 and pos[0]<width and pos[1]<height
    def move1Diagonal(coef):
        pos=(position[0]+coef, position[1]+coef)
        i=coef
        while onBoard(pos):
            yield (i,pos)
            pos=(pos[0]+coef, pos[1]+coef)
            i+=coef
    def move2Diagonal(coef):
        pos=(position[0]+coef, position[1]-coef)
        i=coef
        while onBoard(pos):
            yield (i,pos)
            pos=(pos[0]+coef, pos[1]-coef)
            i+=coef
    pot=list(takewhile(lambda c:board[c[1][0]][c[1][1]]==opponent, move1Diagonal(1)))
    if len(pot)>0:
        val=list(move1Diagonal(pot[-1][0]+1))
        if len(val)>0:
            pos=val[0][1]
            value=board[pos[0]][pos[1]]
            if value==player:
                turn.extend([x[1] for x in pot])

    pot=list(takewhile(lambda c:board[c[1][0]][c[1][1]]==opponent, move1Diagonal(-1)))
    if len(pot)>0:
        val=list(move1Diagonal(pot[-1][0]-1))
        if len(val)>0:
            pos=val[0][1]
            value=board[pos[0]][pos[1]]
            if value==player:
                turn.extend([x[1] for x in pot])

    pot=list(takewhile(lambda c:board[c[1][0]][c[1][1]]==opponent, move2Diagonal(1)))
    if len(pot)>0:
        val=list(move2Diagonal(pot[-1][0]+1))
        if len(val)>0:
            pos=val[0][1]
            value=board[pos[0]][pos[1]]
            if value==player:
                turn.extend([x[1] for x in pot])

    pot=list(takewhile(lambda c:board[c[1][0]][c[1][1]]==opponent, move2Diagonal(-1)))
    if len(pot)>0:
        val=list(move2Diagonal(pot[-1][0]-1))
        if len(val)>0:
            pos=val[0][1]
            value=board[pos[0]][pos[1]]
            if value==player:
                turn.extend([x[1] for x in pot])

    board[position[0]][position[1]]=original
    return turn


def get_opponent(player):
    return PLAYER_1 if player==PLAYER_2 else PLAYER_2

def invert(board):
    return [[PLAYER_1 if j == PLAYER_2 else (PLAYER_2 if j == PLAYER_1 else j) for j in i] for i in board]

def normalize(board, player1_color, player2_color, blank_color):
    mapp = {
        player1_color : PLAYER_1,
        player2_color : PLAYER_2,
        blank_color : EMPTY
    }
    return [[mapp.get(j) for j in i] for i in board]


def get_final_result(board, player):
    '''in case of ended game returns winning player, if there exist any available move, returns false'''
    allStones=[x for row in board for x in row]
    if not PLAYER_1 in allStones:
        return PLAYER_2

    if not PLAYER_2 in allStones:
        return PLAYER_1


    def get_final_score(board):
        allStones=[x for row in board for x in row]
        player1=len([x for x in allStones if x==PLAYER_1])
        player2=len([x for x in allStones if x==PLAYER_2])
        if player1==player2:
            return TIE
        if player1>player2:
            return PLAYER_1
        if player2>player1:
            return PLAYER_2

    if len(allStones)==len(board)*len(board[0]):
        return get_final_score(board)

    if not get_potentials(board, player):
        if not get_potentials(board, get_opponent(player)):
            return get_final_score(board)
    return False
    

def apply_move(board, player, position):
    board_shallow=copy.deepcopy(board)
    board_shallow[position[0]][position[1]]=player
    for change in get_turned_stones(board_shallow, position, player):
        board_shallow[change[0]][change[1]]=player

    return board_shallow

class Board(list):
    def __init__(self, width = None, height = None, iterable = None):
        super(Board, self).__init__([] if iterable is None else iterable)
        
        if not iterable is None:
            self.width=len(iterable)
            self.height=len(iterable[0])
        
        else:
            assert not width is None
            assert not height is None

            self.width=width
            self.height=height
            for _ in range(self.width):
                self.append([0]*self.height)

class CancellationToken:
    def __init__(self):
        self._is_cancelled=False
        self.duration=None
        self.start=None
    def cancel(self):
        self._is_cancelled=True
    def set_timeout(self, duration):
        self.start=time.time()
        self.duration=duration
    def is_cancelled(self):
        if self.duration:
            if (time.time()- self.start) * 1000.0>self.duration:
                self._is_cancelled=True
        return self._is_cancelled