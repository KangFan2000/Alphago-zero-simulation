""" 
Codes from line 46 to 271, the Colour, Tile and Board classes, are from COMP34111, which is the Hex game
MCTS codes from line 350 to 493 are adapted from: https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/mcts_alphaZero.py#L66
The epsilon and environmental feedback parts of MCTS codes are written by hand. 
Recursive part is changed. The use order of expand part is adapted. 
The select part is adapted to try to find some improvement
"""

from ast import Add
from random import choice
from time import sleep
import random
from keras.layers import Conv2D, Dense, Flatten, Input, Reshape
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from collections import deque
from enum import Enum
import wandb
import copy
from keras.regularizers import l2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from keras.initializers import glorot_uniform

# hyper-parameters
LEARNING_RATE = 0.005
device = 'cuda'
VALIDATION_SPLIT=0.2
TEMPERATURE = 0.1
BATCH_SIZE = 20
EPOCHS = 200
UPDATE_TIME = 20
EACH_UPDATE_GAMES = 20
SEARCH_DEPTH = 100
EPSILON = 0.25
EPSILON_DECAY = 0.98
MIN_EPSILON = 0.01
L2 = 0.001
BOARD_SIZE = 9
FIRST_MOVES = 5

# chess game
class Colour(Enum):
    """This enum describes the sides in a game of Hex."""

    # RED is vertical, BLUE is horizontal
    RED = (1, 0)
    BLUE = (0, 1)

    def get_text(colour):
        """Returns the name of the colour as a string."""

        if colour == Colour.RED:
            return "Red"
        elif colour == Colour.BLUE:
            return "Blue"
        else:
            return "None"

    def get_char(colour):
        """Returns the name of the colour as an uppercase character."""

        if colour == Colour.RED:
            return "R"
        elif colour == Colour.BLUE:
            return "B"
        else:
            return "0"

    def from_char(c):
        """Returns a colour from its char representations."""

        if c == "R":
            return Colour.RED
        elif c == "B":
            return Colour.BLUE
        else:
            return None

    def opposite(colour):
        """Returns the opposite colour."""

        if colour == Colour.RED:
            return Colour.BLUE
        elif colour == Colour.BLUE:
            return Colour.RED
        else:
            return None

class Tile:
    """The class representation of a tile on a board of Hex."""

    # number of neighbours a tile has
    NEIGHBOUR_COUNT = 6

    # relative positions of neighbours, clockwise from top left
    I_DISPLACEMENTS = [-1, -1, 0, 1, 1, 0]
    J_DISPLACEMENTS = [0, 1, 1, 0, -1, -1]

    def __init__(self, x, y, colour=None):
        super().__init__()

        self.x = x
        self.y = y
        self.colour = colour

        self.visited = False

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def set_colour(self, colour):
        self.colour = colour

    def get_colour(self):
        return self.colour

    def visit(self):
        self.visited = True

    def is_visited(self):
        return self.visited

    def clear_visit(self):
        self.visited = False

class Board:
    """Class that describes the Hex board."""

    def __init__(self, board_size=BOARD_SIZE):
        super().__init__()

        self._board_size = board_size

        self._tiles = []
        for i in range(board_size):
            new_line = []
            for j in range(board_size):
                new_line.append(Tile(i, j))
            self._tiles.append(new_line)

        self._winner = None

    def from_string(string_input, board_size=BOARD_SIZE, bnf=True):
        """Loads a board from a string representation. If bnf=True, it will
        load a protocol-formatted string. Otherwise, it will load from a
        human-readable-formatted board.
        """

        b = Board(board_size=board_size)

        if (bnf):
            lines = string_input.split(",")
            for i, line in enumerate(lines):
                for j, char in enumerate(line):
                    b.set_tile_colour(i, j, Colour.from_char(char))
        else:
            lines = [line.strip() for line in string_input.split("\n")]
            for i, line in enumerate(lines):
                chars = line.split(" ")
                for j, char in enumerate(chars):
                    b.set_tile_colour(i, j, Colour.from_char(char))

        return b

    def has_ended(self):
        """Checks if the game has ended. It will attempt to find a red chain
        from top to bottom or a blue chain from left to right of the board.
        """

        # Red
        # for all top tiles, check if they connect to bottom
        for idx in range(self._board_size):
            tile = self._tiles[0][idx]
            if (not tile.is_visited() and
                tile.get_colour() == Colour.RED and
                    self._winner is None):
                self.DFS_colour(0, idx, Colour.RED)
        # Blue
        # for all left tiles, check if they connect to right
        for idx in range(self._board_size):
            tile = self._tiles[idx][0]
            if (not tile.is_visited() and
                tile.get_colour() == Colour.BLUE and
                    self._winner is None):
                self.DFS_colour(idx, 0, Colour.BLUE)

        # un-visit tiles
        self.clear_tiles()

        return self._winner is not None

    def clear_tiles(self):
        """Clears the visited status from all tiles."""

        for line in self._tiles:
            for tile in line:
                tile.clear_visit()

    def DFS_colour(self, x, y, colour):
        """A recursive DFS method that iterates through connected same-colour
        tiles until it finds a bottom tile (Red) or a right tile (Blue).
        """

        self._tiles[x][y].visit()

        # win conditions
        if (colour == Colour.RED):
            if (x == self._board_size-1):
                self._winner = colour
        elif (colour == Colour.BLUE):
            if (y == self._board_size-1):
                self._winner = colour
        else:
            return

        # end condition
        if (self._winner is not None):
            return

        # visit neighbours
        for idx in range(Tile.NEIGHBOUR_COUNT):
            x_n = x + Tile.I_DISPLACEMENTS[idx]
            y_n = y + Tile.J_DISPLACEMENTS[idx]
            if (x_n >= 0 and x_n < self._board_size and
                    y_n >= 0 and y_n < self._board_size):
                neighbour = self._tiles[x_n][y_n]
                if (not neighbour.is_visited() and
                        neighbour.get_colour() == colour):
                    self.DFS_colour(x_n, y_n, colour)

    def print_board(self, bnf=True):
        """Returns the string representation of a board. If bnf=True, the
        string will be formatted according to the communication protocol.
        """

        output = ""
        if (bnf):
            for line in self._tiles:
                for tile in line:
                    output += Colour.get_char(tile.get_colour())
                output += ","
            output = output[:-1]
        else:
            leading_spaces = ""
            for line in self._tiles:
                output += leading_spaces
                leading_spaces += " "
                for tile in line:
                    output += Colour.get_char(tile.get_colour()) + " "
                output += "\n"

        return output

    def get_winner(self):
        return self._winner

    def get_size(self):
        return self._board_size

    def get_tiles(self):
        return self._tiles

    def set_tile_colour(self, x, y, colour):
        self._tiles[x][y].set_colour(colour)

# save states to train last move
to_win_states, to_win_mcts_probs = [], []

# handle last move, the last move should have 100%
def get_win_pro(action):
    pros = [0]*(BOARD_SIZE*BOARD_SIZE)
    pros[action] = 1
    return pros

# build model
def build_model():
    initializer = glorot_uniform(seed=0)
    x = Input(shape=(BOARD_SIZE, BOARD_SIZE, 1))
    conv = Conv2D(16, (3, 3), activation='relu', kernel_initializer=initializer, padding='same')(x)
    conv = Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializer, padding='same')(conv)

    # action policy layers
    policy_net = Conv2D(4, (1, 1), activation="relu", kernel_initializer=initializer)(conv)
    policy_net = Flatten()(policy_net)
    policy_net = Dense(BOARD_SIZE*BOARD_SIZE, activation="softmax", kernel_initializer=initializer, kernel_regularizer=l2(L2))(policy_net)

    # state value layers
    value_net = Conv2D(2, (1, 1), activation="relu", kernel_initializer=initializer)(conv)
    value_net = Flatten()(value_net)
    value_net = Dense(32, kernel_initializer=initializer)(value_net)
    value_net = Dense(1, activation="tanh", kernel_initializer=initializer, kernel_regularizer=l2(L2))(value_net)

    model = Model(x, [policy_net, value_net])
    model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(learning_rate=LEARNING_RATE))

    model.summary() # show the framework of model
    return model

test_Model = build_model()
test_Model.save("test_Model.h5")

test_Model_unsupervised = build_model()
test_Model_unsupervised.load_weights("test_Model.h5")

# two methods to calculate probs based on MCTS
def count_probs(visits, temp, search_depth):
    if SEARCH_DEPTH - search_depth <= FIRST_MOVES:
        probs = [i for i in visits]
        probs = probs / np.sum(probs)
    else:
        probs = [i for i in visits]
        max_prob = max(probs)
        probs = [1 if i == max_prob else 0 for i in probs]
        probs = probs / np.sum(probs)
    return probs

# def count_probs(x, temp):
#     x = 1.0/temp * np.log(x)
#     probs = np.exp(x - np.max(x))
#     probs /= np.sum(probs)
#     return probs

# get the state of board at present
def get_this_state(board = Board()):
    this_state = [0]*(BOARD_SIZE*BOARD_SIZE)
    rows = board.print_board().strip().split(",")
    for i, row in enumerate(rows):
        for j, char in enumerate(row):
            if char == 'R':
                this_state[i*BOARD_SIZE + j] = 1  # 'R'为1
            elif char == 'B':
                this_state[i*BOARD_SIZE + j] = -1  # 'B'为-1
            else:
                this_state[i*BOARD_SIZE + j] = 0
    return this_state

# get the legal moves at present
def get_legal_moves(board = Board()):
    avaible_state = [BOARD_SIZE*BOARD_SIZE+1]*(BOARD_SIZE*BOARD_SIZE)
    rows = board.print_board().strip().split(",")
    for i, row in enumerate(rows):
        for j, char in enumerate(row):
            if char == 'R':
                avaible_state[i*BOARD_SIZE + j] = BOARD_SIZE*BOARD_SIZE+1
            elif char == 'B':
                avaible_state[i*BOARD_SIZE + j] = BOARD_SIZE*BOARD_SIZE+1
            else:
                avaible_state[i*BOARD_SIZE + j] = i*BOARD_SIZE + j
    return avaible_state

# MCTS
class TreeNode(object):

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {} # key is action, value is node
        self._n_visits = 0
        self._W = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors, legal_moves):
        # assign action_probs to action_node
        for action in range(BOARD_SIZE*BOARD_SIZE):
            if action not in self._children and action in legal_moves:
                self._children[action] = TreeNode(self, action_priors[0][action])

    # minmax means whether use max layer and min layer in MCTS
    def select(self, c_UCB):
        # if minmax == True:
        return max(self._children.items(), key=lambda child_node: child_node[1].get_U_value(c_UCB))
        # else:
        #     return min(self._children.items(), key=lambda child_node: child_node[1].get_U_value(c_UCB, minmax))

    def update(self, leaf_value):
        self._n_visits += 1
        self._W += leaf_value
        self._Q = self._W / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    # minmax means whether use max layer and min layer in MCTS
    # min layer should use Q - U as UCB to judge force more on smaller score and less frequntly used moves
    def get_U_value(self, c_UCB):
        self._u = c_UCB * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        # if minmax == True:
        return self._Q + self._u
        # else:
        #     return self._Q - self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS(object):
    def __init__(self, policy_value_network, c_UCB=5, n_play_round=SEARCH_DEPTH):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_network
        self._c_UCB = c_UCB
        self._n_play_round = n_play_round

    # one round of MCTS
    def _playout(self, simulate_board, current_player):
        global to_win_states
        global to_win_mcts_probs
        node = self._root
        # self._simulate_board = copy.deepcopy(board)
        # minmax = True
        while True:
            if node.is_leaf():
                break             
            action, node = node.select(self._c_UCB)
            # minmax = not minmax
            simulate_board.set_tile_colour(action//BOARD_SIZE, action%BOARD_SIZE, current_player) 

        state = get_this_state(simulate_board)
        state = np.reshape(state, (1,) + (BOARD_SIZE, BOARD_SIZE, 1))
        action_probs, leaf_value = test_Model.predict(state, verbose=0)

        # get feedback from environment
        end = simulate_board.has_ended()                                 
        legal_moves = get_legal_moves(simulate_board)
        if not end:
            node.expand(action_probs, legal_moves)
        else:
            leaf_value = 1 if simulate_board.get_winner() == current_player else -1
            # record the end state and last move to train the last move
            to_end_state = get_this_state(simulate_board)
            to_end_state[action] = 0
            if to_end_state not in to_win_states:
                to_win_states.append(to_end_state)
                to_win_mcts_probs.append(get_win_pro(action))
                
        node.update_recursive(leaf_value)

    def get_move_probs(self, board, current_player, temp=0.05, search_depth = 25):
        root_state = get_this_state(board)
        root_state = np.reshape(root_state, (1,) + (BOARD_SIZE, BOARD_SIZE, 1))
        root_probs, root_value = test_Model.predict(root_state, verbose=0)
        root_legal = get_legal_moves(board)

        # do MCTS with search_depth as the depth
        self._root.expand(root_probs, root_legal)
        for n in range(search_depth):
            simulate_board = copy.deepcopy(board)
            self._playout(simulate_board, current_player)

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)

        # calculate probility distribution
        act_probs = count_probs(visits, temp, search_depth)                  
         
        return acts, act_probs

    def update_with_move(self):
        self._root = TreeNode(None, 1.0)

class MCTSPlayer(object):
    def __init__(self, policy_value_network, c_UCB=5, n_playout=11):
        self.mcts = MCTS(policy_value_network, c_UCB, n_playout)
        self.epsilon = EPSILON

    def reset_player(self):
        self.mcts.update_with_move()

    # reduce epsilon as training goes on
    def reduce_epsilon(self):
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            print("now epsilon is " + str(self.epsilon))
        else:
            print("final epsilon is " + str(self.epsilon))

    def get_action(self, board, current_player, temp=0.05, search_depth = 25):
        move_probs = np.zeros(BOARD_SIZE*BOARD_SIZE)
        acts, probs = self.mcts.get_move_probs(board, current_player, temp, search_depth)
        move_probs[list(acts)] = probs
        
        # select a move based on move_probs and epsilon
        move = np.random.choice(
            acts,
            p=probs*(1 - self.epsilon) + self.epsilon*np.random.dirichlet(0.3*np.ones(len(probs)))
        )
        
        self.mcts.update_with_move()

        return move, move_probs

# wandb.init(
#     project="wandb_test",
#     config={
#     }
# )


class TrainAgent():

    def __init__(self, board_size=BOARD_SIZE):

        self.board = Board()
        self.this_state = np.zeros(BOARD_SIZE*BOARD_SIZE)
        self.colour = Colour.RED
        self.if_end = False
        self.round = 0
        self.epsilon = EPSILON 

        self.player = MCTSPlayer(test_Model, n_playout=SEARCH_DEPTH)
    
    def opp_colour(self):
        if self.colour == Colour.RED:
            return Colour.BLUE
        elif self.colour == Colour.BLUE:
            return Colour.RED
        else:
            return "None"
        
    def init_board(self):
        self.board = Board()
        self.colour = Colour.RED
        self.if_end = False
        self.this_state = np.zeros(BOARD_SIZE*BOARD_SIZE)
        # self.this_state_input = self.get_state_input()

    def train(self):
        global to_win_states
        global to_win_mcts_probs
        red_win_sum = []
        # states, mcts_probs, current_players = [], [], []
        all_to_win_states, all_to_win_mcts_probs = [], []
        to_win_states, to_win_mcts_probs = [], []
        for m in range(UPDATE_TIME):
            print("train_round " + str(m))
            states, mcts_probs, current_players, winners_z_list = [], [], [], []
            to_win_states, to_win_mcts_probs = [], []
            red_win_count = 0
            for episode in range(EACH_UPDATE_GAMES):
                self.round += 1
                move_count = 0
                self.init_board()
                current_players = []
                # states, mcts_probs, current_players = [], [], []
                while not self.if_end:
                    move_count += 1
                    self.this_state = get_this_state(self.board)
                    # if move_count <= 4 and episode < UPDATE_TIME - 4:
                    #     pos, move_probs = self.player.get_action(self.this_state, self.board, self.colour, temp = 1, search_depth = SEARCH_DEPTH - move_count)
                        # legal_moves = get_legal_moves(self.board)
                        # legal_set = [i for i in legal_moves if i != 122]
                        # pos = random.choice(legal_set)
                        # self.board.set_tile_colour(pos//BOARD_SIZE, pos%BOARD_SIZE, self.colour)
                        # self.if_end = self.board.has_ended()
                        # self.colour = self.opp_colour()
                    # else:
                    pos, move_probs = self.player.get_action(self.board, self.colour, temp = TEMPERATURE, search_depth = SEARCH_DEPTH - move_count)
                    # print("pos " + str(pos))

                    # store the data
                    states.append(self.this_state)
                    mcts_probs.append(move_probs)
                    current_players.append(self.colour)

                    # make move
                    self.board.set_tile_colour(pos//BOARD_SIZE, pos%BOARD_SIZE, self.colour)

                    # if game ends, assign z to each turn
                    self.if_end = self.board.has_ended()
                    if self.if_end:
                        winners_z = np.zeros(len(current_players))
                        winners_z[np.array(current_players) == self.colour] = 1.0
                        winners_z[np.array(current_players) != self.colour] = -1.0
                        winners_z_list.extend(winners_z)
                        self.player.reset_player()
                        if m <= 5 and len(winners_z_list) < 40:
                            print("winners_z: ")
                            print(winners_z_list)
                        # print(self.board.get_winner())
                        # print(self.board.print_board())

                    self.colour = self.opp_colour()
                if self.colour == Colour.BLUE:
                    red_win_count += 1
                    
            # MCTS train
            winners_z_set = np.array(winners_z_list)
            # if m <= 5:
            #     print("winners_z_set[:20]: ")
            #     print(winners_z_set[:20])
            n = len(states)
            print("n: " + str(n))
            flat_list = [item for sublist in states for item in sublist]
            arr = np.array(flat_list)
            re_states = arr.reshape(n, BOARD_SIZE, BOARD_SIZE, 1)
            n = len(mcts_probs)
            flat_list = [item for sublist in mcts_probs for item in sublist]
            arr = np.array(flat_list)
            re_mcts_probs = arr.reshape(n, BOARD_SIZE*BOARD_SIZE, 1)
            test_Model.fit(re_states, [re_mcts_probs, winners_z_set], batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_split=0.2)
            test_Model_unsupervised.fit(re_states, [re_mcts_probs, winners_z_set], batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_split=0.2)
            # with open("history.txt", "a") as file:
            #     for key, value in history.history.items():
            #         file.write(f"{key}: {value}\n")
            # with open("history_unsupervised.txt", "a") as file:
            #     for key, value in history_unsupervised.history.items():
            #         file.write(f"{key}: {value}\n")
            if m > 5:
                model_name = "test_Model_unsupervised_without_final_%d.h5" % m
                test_Model_unsupervised.save(model_name)

            # train with the states which will win in one step
            red_win_sum.append(red_win_count)
            self.player.reduce_epsilon()
            print(red_win_sum)

            all_to_win_states.extend(to_win_states)
            all_to_win_mcts_probs.extend(to_win_mcts_probs)
        
            print()
            n = len(to_win_states)
            print("n: " + str(n))
            flat_list = [item for sublist in to_win_states for item in sublist]
            arr = np.array(flat_list)
            re_states = arr.reshape(n, BOARD_SIZE, BOARD_SIZE, 1)
            n = len(to_win_mcts_probs)
            flat_list = [item for sublist in to_win_mcts_probs for item in sublist]
            arr = np.array(flat_list)
            re_mcts_probs = arr.reshape(n, BOARD_SIZE*BOARD_SIZE, 1)
            to_win_z = np.full(len(to_win_states), -1.0)
            test_Model.fit(re_states, [re_mcts_probs, to_win_z], batch_size=20, epochs=20, validation_split=0.2)
            if m > 5:
                model_name = "test_Model_supervised_include_final_%d.h5" % m
                test_Model.save(model_name)
        # //supervise
        # //validation
            
        test_Model.load_weights("test_Model.h5")
        print()
        n = len(all_to_win_states)
        print("n: " + str(n))
        flat_list = [item for sublist in all_to_win_states for item in sublist]
        arr = np.array(flat_list)
        re_states = arr.reshape(n, BOARD_SIZE, BOARD_SIZE, 1)
        n = len(all_to_win_mcts_probs)
        flat_list = [item for sublist in all_to_win_mcts_probs for item in sublist]
        arr = np.array(flat_list)
        re_mcts_probs = arr.reshape(n, BOARD_SIZE*BOARD_SIZE, 1)
        to_win_z = np.full(len(all_to_win_states), -1.0)
        test_Model.fit(re_states, [re_mcts_probs, to_win_z], batch_size=20, epochs=20, validation_split=0.2)
        test_Model.save("test_Model_only_final.h5")

        test_Model_unsupervised.fit(re_states, [re_mcts_probs, to_win_z], batch_size=20, epochs=20, validation_split=0.2)
        test_Model_unsupervised.save("test_Model_unsupervised_with_final.h5")
            

if (__name__ == "__main__"):
    agent = TrainAgent()
    agent.train()
    # wandb.finish()
