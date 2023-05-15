import numpy as np

from part_5 import *


initial_board = [-1,-1,-1,-1,0,0,0,1,1,1]

class HexaPawn:
    def __init__(self, board):
        self.board = list(board)

    def to_move(self):
        return self.board[0]

    def actions(self):
        actions = []
        if self.board[0] == 1:  # white player's turn
            for i in range(1, len(self.board)):
                if self.board[i] == 1:  # white pawn can move up
                    if i-3 >= 0 and self.board[i-3] == 0:
                        actions.append(('advance', i, i-3))
                    if (i-1)%3 != 0 and i-4 >= 0 and self.board[i-4] == -1:
                        actions.append(('capture-left', i, i-4))
                    if (i-1)%3 != 2 and i-2 >= 0 and self.board[i-2] == -1:
                        actions.append(('capture-right', i, i-2))
        else:  # black player's turn
            for i in range(1, len(self.board)):
                if self.board[i] == -1:  # black pawn can move down
                    if i+3 <= 9 and self.board[i+3] == 0: #advance 
                        actions.append(('advance', i, i+3))
                    if (i-1)%3 != 0 and i+2 <= 9 and self.board[i+2] == 1: #capture left 
                        actions.append(('capture-left', i, i+2))
                    if (i-1)%3 != 2 and i+4 <= 9 and self.board[i+4] == 1: #capture right 
                        actions.append(('capture-right', i, i+4))
        return actions

    def result(self, action):
        new_board = self.board.copy()
        new_board[action[2]] = new_board[action[1]]
        new_board[action[1]] = 0
        new_board[0] = -new_board[0]
        return HexaPawn(new_board)

    def is_terminal(self):
        if self.utility():
            return True
        return False

    def utility(self):
        for i in range(1, 4):
            if self.board[i] == 1:
                return 1
        for j in range(7, 10):
            if self.board[j] == -1:
                return -1
        if not self.actions():
            return -self.board[0]
        return 0

def minimax_search(game):
    value , move = max_value(game)
    return move

def max_value(game):
    if game.is_terminal():
        return game.utility(), None
    v = -float('inf')
    move = None
    for a in game.actions():
        v2 , a2 = min_value(game.result(a))
        if v2 > v:
            v = v2
            move = a
    return v, move
        

def min_value(game):
    if game.is_terminal():
        return game.utility(), None
    v = float('inf')
    move = None
    for a in game.actions():
        v2 , a2 = max_value(game.result(a))
        if v2 < v:
            v = v2
            move = a
    return v, move



def build_policy_table():
    Xs, ys = [], []
    for sp in getAllStates(initial_board):
        game = HexaPawn(sp)
        act = minimax_search(game)
        next_sp = game.result(act)
        Xs.append(game.board)
        ys.append(next_sp.board[1:])
    return Xs, ys

def getAllStates(board):
    initalGame = HexaPawn(board)
    stateSpace = set()
    stateQueue =  [initalGame]
    while stateQueue:
        s = stateQueue.pop()
        # print(s.board)
        if s.is_terminal():
            continue
        acts = s.actions()
        stateSpace.add(tuple(s.board))
        for a in acts:
            stateQueue.append(s.result(a))  
    return stateSpace 

def train_neural_network(x, y):
    w1 = np.random.uniform(-1, 1, size = (10,16))
    
    b1 = np.random.uniform(-1, 1, size = (1,16))

    w2 = np.random.uniform(-1, 1, size = (16,9))
    b2 = np.random.uniform(-1, 1, size = (1,9))

    for i in range(1000):
        dW1 , dW2 , dB1, dB2 = classify(np.array(x),np.array(y),w1,w2,b1,b2)
        w1,w2,b1,b2 = update_weights(w1, w2, b1, b2, dW1, dW2, dB1, dB2)

    for _ in range(10):
        idx = np.random.randint(0,len(x))
        y_pred = predict(np.array(x[idx]),w1,w2,b1,b2)
        print("Input: ",x[idx])
        print("Output: ",y[idx])
        print("Predicted: ",y_pred)

hexBoard = HexaPawn(initial_board)

X, y = build_policy_table()

print(len(X[0]))
print(len(y[0]))

train_neural_network(X, y)
