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

    row_index = sum(board[:, action] == 0) - 1 # Get all row dimensions for the action column to take. I.E. get how many rows are 0 for column index 3?
    # If all empty, there are 6 slots. So sum 6, -1 = row index 5. Piece will appear on the 3,5 which is the last row 4th column
    board[row_index, action] = player # Update board if player made move +1, if opponent, -1
    return board

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