from constants import *
from network import *

# Part 1, 2

class HexaPawn:                                                             # class HEXAPAWN
    def __init__(self, board):                                              # function INIT
        self.board = list(board)                                            # initialize board          

    def to_move(self):                                                      # function TO_MOVE
        return self.board[0]                                                # return player to move

    def actions(self):                                                      # function ACTIONS                    
        actions = []                                                        # initialize actions                
        if self.board[0] == 1:                                              # check if it's white player's turn
            for i in range(1, len(self.board)):                             # iterate through the board
                if self.board[i] == 1:                                      # check if it's a white pawn
                    if i-3 >= 0 and self.board[i-3] == 0:                   # check if it can advance
                        actions.append(('advance', i, i-3))                 # add advance action to actions
                    if (i-1)%3 != 0 and i-4 >= 0 and self.board[i-4] == -1: # check if it can capture left
                        actions.append(('capture-left', i, i-4))            # add capture left action to actions
                    if (i-1)%3 != 2 and i-2 >= 0 and self.board[i-2] == -1: # check if it can capture right
                        actions.append(('capture-right', i, i-2))           # add capture right action to actions
        else:                                                               # black player's turn
            for i in range(1, len(self.board)):                             # iterate through the board
                if self.board[i] == -1:                                     # black pawn can move down
                    if i+3 <= 9 and self.board[i+3] == 0:                   # check if it can advance
                        actions.append(('advance', i, i+3))                 # add advance action to actions
                    if (i-1)%3 != 0 and i+2 <= 9 and self.board[i+2] == 1:  # check if it can capture left
                        actions.append(('capture-left', i, i+2))            # add capture left action to actions
                    if (i-1)%3 != 2 and i+4 <= 9 and self.board[i+4] == 1:  # check if it can capture right
                        actions.append(('capture-right', i, i+4))           # add capture right action to actions
        return actions                                                      # return actions

    def result(self, action):                                               # function RESULT
        new_board = self.board.copy()                                       # initialize new board
        new_board[action[2]] = new_board[action[1]]                         # move pawn
        new_board[action[1]] = 0                                            # remove pawn from old position
        new_board[0] = -new_board[0]                                        # change player to move
        return HexaPawn(new_board)                                          # return new game state   

    def is_terminal(self):                                                  # function IS_TERMINAL        
        if self.utility():                                                  # check if game is over       
            return True                                                     # return True if game is over            
        return False                                                        # return False if game is not over

    def utility(self):                                                      # function UTILITY                
        for i in range(1, 4):                                               # check if white player won
            if self.board[i] == 1:                                          # check if it's a white pawn    
                return 1                                                    # return 1 if white player won
        for j in range(7, 10):                                              # check if black player won
            if self.board[j] == -1:                                         # check if it's a black pawn
                return -1                                                   # return -1 if black player won
        if not self.actions():                                              # check if game is draw
            return -self.board[0]                                          
        return 0                                                            