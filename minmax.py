# Part 2 
from hexapawn import HexaPawn
from constants import *

def minimax_search(game):                       # function MINIMAX-SEARCH(game, state) returns an action
    value , move = max_value(game)              # value, move ← MAX-VALUE(game, state)
    return move                                 # return move

def max_value(game):                            # function MAX-VALUE(game, state) returns a (utility, move) pair
    if game.is_terminal():                      # if game.IS-TERMINAL(state) then 
        return game.utility(), None             # return game.UTILITY(state, player ), null
    v = -float('inf')                           # v ← −∞
    move = None                                 # 
    for a in game.actions():                    # for each a in game.ACTIONS(state) do
        v2 , a2 = min_value(game.result(a))     # v2 , a2 ← MIN-VALUE(game, game.RESULT(state, a))
        if v2 > v:                              # if v2 > v
            v = v2                              # then v ← v2
            move = a                            # move ← a
    return v, move                              # return v, move

def min_value(game):                            # function MIN-VALUE(game, state) returns a (utility, move) pair
    if game.is_terminal():                      # if game.IS-TERMINAL(state) then 
        return game.utility(), None             # return game.UTILITY(state, player ), null 
    v = float('inf')                            # v ← +∞
    move = None                                 #               
    for a in game.actions():                    # for each a in game.ACTIONS(state) do
        v2 , a2 = max_value(game.result(a))     # v2 , a2 ← MAX-VALUE(game, game.RESULT(state, a))
        if v2 < v:                              # if v2 < v then
            v = v2                              # v ← v2
            move = a                            # move ← a
    return v, move                              # return v, move

def getAllStates(board):                        # function to get all possible states from a given board
    initalGame = HexaPawn(board)                # initialize game
    stateSpace = set()                          # initialize state space
    stateQueue =  [initalGame]                  # initialize state queue
    while stateQueue:                           # while state queue is not empty
        s = stateQueue.pop()                    # pop state from queue
        if s.is_terminal():                     # if state is terminal, continue
            continue
        acts = s.actions()                      # get all possible actions from state
        stateSpace.add(tuple(s.board))          # add state to state space
        for a in acts:                          # for each action in actions
            stateQueue.append(s.result(a))      # add result of action to state queue
    return stateSpace                           # return state space

def build_policy_table():                       # function BUILD-POLICY-TABLE() returns a policy table
    Xs, ys = [], []                             # Xs ← [], ys ← []
    for sp in getAllStates(initial_board):      # for each state in all the possible states do
        game = HexaPawn(sp)                     # game = HexaPawn(state)
        act = minimax_search(game)              # get move to do as per minimax search
        next_sp = game.result(act)              # get result if move is taken as per minimax search
        Xs.append(game.board)                   # append current board to Xs
        ys.append(next_sp.board[1:])            # append next board to ys
    return Xs, ys                               # return Xs, ys



