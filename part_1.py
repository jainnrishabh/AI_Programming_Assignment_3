initial_board = [-1,-1,-1,-1,0,0,0,1,1,1]

class HexaPawn:
    def __init__(self, board):
        self.board = list(board)

    def to_move(self):
        if self.board[0] == -1 :
            return 1
        else :
            return -1

    def actions(self):
        actions = []
        if self.board[0] == 1:  # white player's turn
            for i in range(len(self.board)):
                if self.board[i] == 1:  # white pawn can move up
                    if i-3 >= 0 and self.board[i-3] == 0:
                        actions.append((i, i-3))
                    if i-2 >= 0 and self.board[i-2] == -1:
                        actions.append((i, i-2))
                    if i-1 >= 0 and self.board[i-1] == -1:
                        actions.append((i, i-1))
        else:  # black player's turn
            for i in range(len(self.board)):
                if self.board[i] == -1:  # black pawn can move down
                    if i+3 <= 8 and self.board[i+3] == 0: #advance 
                        actions.append((i, i+3))
                    if i+2 <= 8 and self.board[i+2] == 1: #capture left 
                        actions.append((i, i+2))
                    if i+1 <= 8 and self.board[i+1] == 1: #capture right 
                        actions.append((i, i+1))
        return actions

    def result(self, action):
        new_board = self.board.copy()
        new_board[action[1]] = new_board[action[0]]
        new_board[action[0]] = 0
        return HexaPawn(new_board)

    def is_terminal(self):
        if self.board[0] == 1:
            return all([p == -1 for p in self.board])
        else:
            return all([p == 1 for p in self.board])

    def utility(self, player):
        if self.board[0] == 1:  # white player won
            if all([p == -1 for p in self.board]):
                return 1
            else:
                return -1
        else:  # black player won
            if all([p == 1 for p in self.board]):
                return 1
            else:
                return -1

def minimax(state, player, depth, alpha, beta):
    if depth == 0 or state.is_terminal():
        return state.utility(player), None
    
    best_action = None
    if state.to_move == player:
        best_value = -float('inf')
        for action in state.actions():
            result = state.result(action)
            value, _ = minimax(result, player, depth-1, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break
    else:
        best_value = float('inf')
        for action in state.actions():
            result = state.result(action)
            value, _ = minimax(result, player, depth-1, alpha, beta)
            if value < best_value:
                best_value = value
                best_action = action
            beta = min(beta, best_value)
            if beta <= alpha:
                break
                
    return best_value, best_action


def build_policy_table(player, depth):
    policy = set()
    for state in getAllStates(initial_board):
        state = HexaPawn(state)
        for action in state.actions():
            result = state.result(action)
            value, _ = minimax(result, player, depth, -float('inf'), float('inf'))
            if tuple(state.board) not in policy:
                policy.add(tuple(state.board))
            
    return policy

def getAllStates(board):
    initalGame = HexaPawn(board)
    stateSpace = set()
    stateQueue =  [initalGame]
    while stateQueue:
        s = stateQueue.pop()
        if s.is_terminal():
            continue
        acts = s.actions()
        stateSpace.add(tuple(s.board))
        for a in acts:
            stateQueue.append(s.result(a))  
    return stateSpace 

hexBoard = HexaPawn(initial_board)

policy = build_policy_table(1, 100)

print(len(policy))
