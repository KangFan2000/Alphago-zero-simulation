""" 
Colour, Tile and Board classes are from COMP34111, which is the Hex game.
Other codes are written by hand.
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

MODEL_NAME = "test_Model_unsupervised_without_final_14.h5"
TEST_TIMES = 500
L2 = 0.001
LEARNING_RATE = 0.005
BOARD_SIZE = 9

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

    def __init__(self, board_size=5):
        super().__init__()

        self._board_size = board_size

        self._tiles = []
        for i in range(board_size):
            new_line = []
            for j in range(board_size):
                new_line.append(Tile(i, j))
            self._tiles.append(new_line)

        self._winner = None

    def from_string(string_input, board_size=5, bnf=True):
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

# get the board at present
def get_this_state(board = Board()):
    legal_moves = []
    this_state = [0]*BOARD_SIZE*BOARD_SIZE
    rows = board.print_board().strip().split(",")
    for i, row in enumerate(rows):
        for j, char in enumerate(row):
            if char == 'R':
                this_state[i*BOARD_SIZE + j] = 1  # 'R'为1
            elif char == 'B':
                this_state[i*BOARD_SIZE + j] = -1  # 'B'为-1
            else:
                this_state[i*BOARD_SIZE + j] = 0
                legal_moves.append(i*BOARD_SIZE + j)
    return this_state, legal_moves

def get_legal_moves(board = Board()):
    avaible_state = [BOARD_SIZE^2+1]*(BOARD_SIZE*BOARD_SIZE)
    rows = board.print_board().strip().split(",")
    for i, row in enumerate(rows):
        for j, char in enumerate(row):
            if char == 'R':
                avaible_state[i*BOARD_SIZE + j] = BOARD_SIZE^2+1
            elif char == 'B':
                avaible_state[i*BOARD_SIZE + j] = BOARD_SIZE^2+1
            else:
                avaible_state[i*BOARD_SIZE + j] = i*BOARD_SIZE + j
    return avaible_state

class NaiveAgent():

    def __init__(self, board_size=BOARD_SIZE):

        self.board = Board()
        self.this_state = np.zeros(BOARD_SIZE*BOARD_SIZE)
        self.colour = Colour.RED
        self.if_end = False
        self.round = 0
    

    def opp_colour(self):
        if self.colour == Colour.RED:
            return Colour.BLUE
        elif self.colour == Colour.BLUE:
            return Colour.RED
        else:
            return "None"
        
    def init_board(self):
        self.board = Board(BOARD_SIZE)
        self.colour = Colour.RED
        self.if_end = False
        self.this_state = np.zeros(BOARD_SIZE*BOARD_SIZE)

    def test(self):
        test_Model.load_weights(MODEL_NAME)
        model_win_count = 0
        model_loss_count = 0
        T = True
        for m in range(TEST_TIMES):
            self.round += 1
            self.init_board()
            # states, mcts_probs, current_players = [], [], []

            # play a game
            while not self.if_end:
                this_state, legal_moves = get_this_state(self.board)
                this_state = np.reshape(this_state, (1,) + (BOARD_SIZE, BOARD_SIZE, 1))

                # model turn
                pos_probs, val = test_Model.predict(this_state, verbose=0)
                combined_data = [(pos_probs[0][i], i) for i in legal_moves]
                sorted_data = sorted(combined_data, key=lambda x: x[0], reverse=True)
                pos_index = sorted_data[0][1]
                pos = [pos_index//BOARD_SIZE, pos_index%BOARD_SIZE]

                # make move
                self.board.set_tile_colour(pos[0], pos[1], self.colour)
                if T:
                    print(pos)

                # if game ends
                self.if_end = self.board.has_ended()
                if self.if_end:
                    model_win_count += 1
                    T = False
                    break

                self.colour = self.opp_colour()
                
                # random turn
                _, legal_moves = get_this_state(self.board)
                pos = choice(legal_moves)
                self.board.set_tile_colour(pos//BOARD_SIZE, pos%BOARD_SIZE, self.colour)
                if T:
                    print(str(pos//BOARD_SIZE) + "," + str(pos%BOARD_SIZE))

                # if game ends
                self.if_end = self.board.has_ended()
                if self.if_end:
                    model_loss_count += 1
                    T = False
                    break

                self.colour = self.opp_colour()


        print(MODEL_NAME)
        print("final win times: " + str(model_win_count))
            

if (__name__ == "__main__"):
    agent = NaiveAgent()
    agent.test()
    # wandb.finish()
