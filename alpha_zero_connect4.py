import numpy as np

#1. get_init_board get board
#2. Place pieces
#3. Get Valid moves
#4. is board full
#5 is win

ROW_COUNT = 6 # Num of rows
COL_COUNT = 7 # Num of columns

def get_init_board():
    """
    Initialize Connect 4 Board
    :return: Board represented as numpy array 7 length, 6 height or width
    """
    return np.zeros((ROW_COUNT,COL_COUNT))

def place_piece(board, player, action):
    """
    Action taken by 1 of 2 players. Dropping a piece into a board
    :param board: The connect 4 board at its current state
    :param player: The player making the move
    :param action: The action taken by the player
    :return:
    """
    # Connect 4 works by have pieces fall in the next unoccupied slot
    # If there's a piece, we find the slot above if available. That will be our free slot
    board_copy = np.copy(board)
    row_index = sum(board_copy[:, action] == 0) - 1 # Get all row dimensions for the action column to take. I.E. get how many rows are 0 for column index 3?
    # If all empty, there are 6 slots. So sum 6, -1 = row index 5. Piece will appear on the 3,5 which is the last row 4th column
    board_copy[row_index, action] = player # Update board if player made move +1, if opponent, -1
    return board_copy

def get_valid_moves(board):
    # Return list of valid moves on the board
    # return [0,1,1,1,0,1,1] 0 is invalid 1 is valid
    valid_moves = [0] * COL_COUNT
    for column in range(COL_COUNT):
        if sum(board[:,column] == 0) > 0:
            valid_moves[column] = 1

    return valid_moves

def is_board_full(board):
    """
    Check if there's no spaces to place on the board
    :param board:
    :return:
    """
    return sum(board.flatten()==0) == 0

def is_win(board, player):
    """
    Check if player or other player won with pieces following
    Top left to bottom left
    Bottom left to top left
    Vertical
    Row
    :param board: Connect 4 Board
    :param player: Player or Opponent
    :return: True if player won else False
    """
    # Vertical Win
    for column in range(COL_COUNT):
        # for row in range(ROW_COUNT):
        #     return board[column, range(row,row+4)] == player
        for row in range(3): # go from 2nd index
            if board[row, column] == board[row+1, column] == board[row+2, column] == board[row+3, column] == player:
                return True

    # Horizontal Win
    for row in range(ROW_COUNT):
        for column in range(4): # 4 Elements not index
            if board[row, column] == board[row, column+1] == board[row, column+2] == board[row, column+3] == player:
                return True

    # Diagonal top left to bottom right
    for row in range(3): # Don't need to check whole board, only portion that can fit 4 diagonals
        for column in range(4):
            if board[row, column] == board[row+1, column+1] == board[row+2,column+2] == board[row+3, column+3] == player:
                return True
    # Diagonal bottom left to top right
    for row in range(5,2,-1): # Don't need to check whole board, only portion that can fit 4 diagonals
        for column in range(4):
            if board[row, column] == board[row-1, column+1] == board[row-2,column+2] == board[row-3, column+3] == player:
                return True


    return False

if __name__ == '__main__':
    board = get_init_board()
    board = place_piece(board, player=1, action=3)
    board = place_piece(board, player=1, action=3)
    board = place_piece(board, player=1, action=3)
    board = place_piece(board, player=1, action=3)
    board = place_piece(board, player=1, action=3)
    board = place_piece(board, player=1, action=3)
    print(board)
    print(get_valid_moves(board))