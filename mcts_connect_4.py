from alpha_zero_connect4 import get_init_board, place_piece, get_valid_moves, is_board_full, is_win
import numpy as np
import random
import math

# How to generate good training data so algorithm plays against self and get better as time progresses

NUM_SIMULATION = 100

BOARD = np.array(
    [[0,-1,-1,-1,1,0,-1],
     [0, 1,-1, 1,1,0, 1],
     [-1,1,-1,1,1, 0,-1],
     [1,-1,1,-1,-1,0,-1],
     [-1,-1,1,-1,1,1,-1],
     [-1,1,1,-1,1,-1, 1]]
)


def ucb_score(parent, child):
    prior_score = child.parent * math.sqrt(parent.visits) / (child.visits + 1)
    if child.visits > 0:
        value_score = child.value / child.visits
    else:
        value_score = 0

    return value_score * prior_score # balance exploration with exploitation

def dummy_model_predict(board):
    # Policy Head return action probability
    # Value Head return a value that is associated with how strong the board position is for player 1
    value_head = 0.5 # 0 - 1
    policy_head = [0.5, 0, 0, 0, 0, 0.5, 0]
    return value_head, policy_head

class Node:
    def __init__(self, parent, turn, state):
        # Parent Node
        # Which turn it is, self or opponent
        # Board State
        self.parent = parent
        self.turn = turn
        self.state = state
        self.children = {}
        self.value = 0
        self.visits = 0

    def expand(self, action_probs):
        # List of values with probabilities [0.5, 0, 0, 0, 0, 0.5, 0] Change network selects the choice
        for action, prob in enumerate(action_probs):
            if prob > 0:
                next_state = place_piece(board=self.state, player=self.turn, action=action)
                # Create children node from non 0. Also indicate next turn is made by opponent
                self.children[action] = Node(parent=prob, turn=self.turn * -1, state=next_state)

    def select_child(self):
        # UCB Score
        max_score = -99
        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > max_score:
                selected_action = action
                selected_child = child
                max_score = score

        return selected_action, selected_child


if __name__ == '__main__':
    # Player 1 = 1
    # Opponent = -1
    # initialize root
    root = Node(parent=None, turn=1, state=BOARD)

    # expand the root
    # Create children for all different possible placement that can be made by the player
    value, action_probs = dummy_model_predict(root.state)
    root.expand(action_probs=action_probs)

    # print(root.children[0].state)
    # print(root.children[5].state)

    # Iterate through simulation
    for _ in range(NUM_SIMULATION):
        node = root
        search_path = [node]
        # Select Next child until we reach an unexpected node
        while len(node.children) > 0:
            action, node = node.select_child()
            # Append to search path
            search_path.append(node)
        value = None
        # Calculate value once we reach a leaf node
        if is_board_full(node.state):
            value = 0
        if is_win(node.state, player=1):
            value = 1
        if is_win(node.state, player=-1):
            value = -1
        if value is None:
            # If game is not over, get value from network and expand
            value, action_probs = dummy_model_predict(node.state)
            node.expand(action_probs)

        # Back up the value. Check all nodes we touched during the simulation
        for node in search_path:
            node.value += value
            node.visits += 1

    print(root.children[0].state)
    print(root.children[0].value)

    print(root.children[5].state)
    print(root.children[5].value)

    # run simulations