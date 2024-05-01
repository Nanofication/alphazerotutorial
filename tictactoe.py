import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import random

from tqdm.notebook import trange

torch.manual_seed(0)

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        # resBlocks number of blocks in state
        # num_hidden - hidden layer node

        super().__init__()
        self.device = device

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)] # Array of different rest blocks
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )

        self.to(self.device)

    def forward(self,x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)

        policy = self.policyHead(x)
        value = self.valueHead(x)

        return policy, value

# Convolution is math operation help extract features from an image like edges, shapes or patterns

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1) # 2D Convolution operation on input image or tensor
        self.bn1 = nn.BatchNorm2d(num_hidden) #
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x): # Is input
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)

        return x

class Node:
    # Define Monte Carlo Tree Search Node
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        # Game and arguments from MCTS
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior # Probability that was initiated in the child node. Probability to land if go down from parent

        self.children = [] # children of the node
        # self.expandable_moves = game.get_valid_moves(state) #  Only for MCTS Which ways to expand further from the node? Next steps to take from current point

        self.visit_count = visit_count
        self.value_sum = 0 # Later for UCB method MCTS's main calculation

    def is_fully_expanded(self):
        # return np.sum(self.expandable_moves) == 0 and len(self.children) > 0 # MCTS
        return len(self.children) > 0 # For Alpha Zero Deep Q Learning

    def select(self):
        """
        Apply UCB Score
        :return:
        """
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            # ucb = self.get_ucb(child)
            ucb = self.get_ucb_alphazero(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        """
        Q(s,a) + C*sqrt(ln(n(s))/n(s,a)
        In TicTacToe/Chess/etc, child node is usually the opponent
        :param child: Child Node
        :return:
        """
        q_value = 1 - ((child.value_sum / child.visit_count) + 1)/2
        # Find opponent's value and if it opponent makes bad move, it's very good for us player so 1 - (next move)

        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)

    def get_ucb_alphazero(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2

        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior

    def expand(self):
        """
        How expand, sample possible moves. Create new state for our parent
        Append to list of children nodes
        :return:
        """
        action = np.random.choice(np.where(self.expandable_moves == 1)[0]) # Randomly pick value out of the array of available actions. Get indices
        self.expandable_moves[action] = 0

        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1) # We will never change the player. Never give node info of player. Change state of child
        # Perceive the opponent

        child_state = self.game.change_perspective(child_state, player=-1) # Assume you're playing against opponent
        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)

        return child

    def expandDql(self, policy):
        """
        How expand, sample possible moves. Create new state for our parent
        Append to list of children nodes
        :return:
        """
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1) # We will never change the player. Never give node info of player. Change state of child
                # Perceive the opponent

                child_state = self.game.change_perspective(child_state, player=-1) # Assume you're playing against opponent
                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

        return child # you can delete this?

    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken) # Have to flip to opponent perspective
        value = self.game.get_opponent_value(value)

        if is_terminal:
            return value

        rollout_state = self.state.copy()
        rollout_player = 1

        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0]) # Pick the first one
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)

            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value

            rollout_player = self.game.get_opponent(rollout_player)

    def backpropogate(self,value):
        # Go all the way up to parent
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropogate(value)

class MCTS:

    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        # Define root node
        root = Node(self.game, self.args, state, visit_count=1)

        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
                                    * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expandDql(policy)

        for search in range(self.args['num_searches']):
            node = root
            # selection
            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                # Use Model
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0) # turn into a tensor
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy() # apply on input of 9 neurons (tic tac toe inputs) Comment to try GPU
                # policy = torch.softmax(policy, axis=1).squeeze(0).numpy()
                valid_moves = self.game.get_valid_moves(node.state) # Get all valid moves
                policy *= valid_moves #
                policy /= np.sum(policy) # Rescale policy each number to percentages

                value = value.item() # Use for backpropogation

                # expansion MCTS
                # node = node.expand()
                node = node.expandDql(policy)


                # simulation MCTS
                # value = node.simulate()
            # backpropogation
            node.backpropogate(value)

        # After done a lot, return number of visit count distribution. Visit counts for the children of our root nodes
        action_probs = np.zeros(self.game.action_size)

        for child in root.children:
            action_probs[child.action_taken] = child.visit_count

        action_probs /= np.sum(action_probs)
        return action_probs

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state

    def get_valid_moves(self, state):
        """
        Any squares where its set to 0 means the move is open
        Any moves where it has 1 or -1 means it's taken
        :param state: Current board's state
        :return:
        """
        return (state.reshape(-1)==0).astype(np.uint8)

    def check_win(self, state, action):
        if action == None: # In root node, no action taken so no one wins
            return False

        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]

        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        """
        Opponent's move is the opposite of players
        :param player: The player's number representation
        :return:
        """
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state ==0, state == 1)
        ).astype(np.float32)

        return encoded_state

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        # First call MCTS
        # Get action probabilities
        # Set Action
        # Use Action to play, get a new state
        # Check if this state is terminal or not

        while True:
            # when call MCTS we're always player 1
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args['temperature'])# Sometimes we want to explore more over exploit. Higher to infinity, more you just choose a random action
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs) # Select action based on probability. Sometimes we want to explore more

            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action) # Get updated state and action

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx: min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample) # Transpose our list. * basically transposes the 3 lists

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def learn(self):
        # Self play, get data, train, self play again
        for iteration in range(self.args['num_iterations']):
            # Create memory class
            memory = []

            self.model.eval()
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in range(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    tictactoe = TicTacToe()
    # model = ResNet(tictactoe, 4, 64, device)
    # # model = model.to() #device=device
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    #
    # args = {
    #     'C': 2,
    #     'num_searches': 60,
    #     'num_iterations': 3,
    #     'num_selfPlay_iterations': 500,
    #     'num_epochs': 4,
    #     'batch_size': 64,
    #     'temperature': 1.25,
    #     'dirichlet_epsilon':  0.25,
    #     'dirichlet_alpha': 0.3
    #
    # }
    #
    # alphaZero = AlphaZero(model, optimizer, tictactoe, args)
    # alphaZero.learn()

    # tictactoe = TicTacToe()
    # player = 1
    #
    # args = {
    #     'C': 2,
    #     'num_searches': 1000
    # }
    #
    # model = ResNet(tictactoe, 4, 64)
    # model.eval()
    #
    # mcts = MCTS(tictactoe, args, model)
    #
    # state = tictactoe.get_initial_state()
    #
    # while True:
    #     print(state)
    #
    #     if player == 1:
    #         valid_moves = tictactoe.get_valid_moves(state)
    #         print("valid moves", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
    #         action = int(input(f"{player}:"))
    #
    #         if valid_moves[action] == 0:
    #             print("action not valid")
    #             continue
    #     else:
    #         neutral_state = tictactoe.change_perspective(state, player)
    #         mcts_probs = mcts.search(neutral_state)
    #         action = np.argmax(mcts_probs) # return child
    #
    #     state = tictactoe.get_next_state(state, action, player)
    #
    #     value, is_terminal = tictactoe.get_value_and_terminated(state, action)
    #
    #     if is_terminal:
    #         print(state)
    #         if value == 1:
    #             print(player, "Won!")
    #         else:
    #             print("Draw")
    #         break
    #
    #     player = tictactoe.get_opponent(player)



    state = tictactoe.get_initial_state()
    state = tictactoe.get_next_state(state, 2, -1)
    state = tictactoe.get_next_state(state, 4, -1)
    state = tictactoe.get_next_state(state, 6, 1)
    state = tictactoe.get_next_state(state, 8, 1)

    encoded_state = tictactoe.get_encoded_state(state) # Have states where player played, where opponent played and where no plays have been done yet

    # Turn state to tensor

    tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0) # Create further bracket on encoded state and pass through model

    model = ResNet(tictactoe, 4, 64, device=device)
    model.load_state_dict(torch.load('model_2.pt')) # Load trained model
    model.eval()
    print(state)
    policy, value = model(tensor_state)
    value = value.item()
    policy = torch.softmax(policy, axis=1).squeeze(0).cpu().detach().numpy() # don't apply on other axis

    print(value, policy)

    plt.bar(range(tictactoe.action_size), policy)
    plt.show()

    # MCTS Example
    # tictactoe = TicTacToe()
    # player = 1
    #
    # args = {
    #     'C': 2,
    #     'num_searches': 1000
    # }
    #
    # mcts = MCTS(tictactoe, args)
    # state = tictactoe.get_initial_state()
    #
    # while True:
    #     print(state)
    #
    #     if player == 1:
    #         valid_moves = tictactoe.get_valid_moves(state)
    #         print("valid moves", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
    #         action = int(input(f"{player}:"))
    #
    #         if valid_moves[action] == 0:
    #             print("action not valid")
    #             continue
    #     else:
    #         neutral_state = tictactoe.change_perspective(state, player)
    #         mcts_probs = mcts.search(neutral_state)
    #         action = np.argmax(mcts_probs) # return child
    #
    #     state = tictactoe.get_next_state(state, action, player)
    #
    #     value, is_terminal = tictactoe.get_value_and_terminated(state, action)
    #
    #     if is_terminal:
    #         print(state)
    #         if value == 1:
    #             print(player, "Won!")
    #         else:
    #             print("Draw")
    #         break
    #
    #     player = tictactoe.get_opponent(player)