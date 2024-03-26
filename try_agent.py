""" 
This codes are adapted from COMP34111. An network is added and how to make move is changed. This agent makes move based on network.
"""

import socket
from random import choice
from time import sleep
from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, Reshape
from keras.regularizers import l2
from keras.optimizers import Adam
import numpy as np
from keras.initializers import glorot_uniform

BOARD_SIZE = 9
LEARNING_RATE = 0.005
L2 = 0.001

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


class NaiveAgent():
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.
    """

    HOST = "127.0.0.1"
    PORT = 1234
    

    def __init__(self, board_size=11):
        self.s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )

        self.s.connect((self.HOST, self.PORT))

        self.board_size = board_size
        self.board = []
        self.colour = ""
        self.turn_count = 0
        self.test_Model = build_model()
        self.test_Model.load_weights("test_Model_unsupervised_without_final_14.h5")

    def run(self):
        """Reads data until it receives an END message or the socket closes."""

        while True:
            data = self.s.recv(1024)
            if not data:
                break
            # print(f"{self.colour} {data.decode('utf-8')}", end="")
            if (self.interpret_data(data)):
                break

        # print(f"Naive agent {self.colour} terminated")

    def interpret_data(self, data):
        """Checks the type of message and responds accordingly. Returns True
        if the game ended, False otherwise.
        """

        messages = data.decode("utf-8").strip().split("\n")
        messages = [x.split(";") for x in messages]
        # print(messages)
        for s in messages:
            if s[0] == "START":
                self.board_size = int(s[1])
                self.colour = s[2]
                self.board = [
                    [0]*self.board_size for i in range(self.board_size)]

                if self.colour == "R":
                    self.make_move()

            elif s[0] == "END":
                return True

            elif s[0] == "CHANGE":
                if s[3] == "END":
                    return True

                elif s[1] == "SWAP":
                    self.colour = self.opp_colour()
                    if s[3] == self.colour:
                        self.make_move()

                elif s[3] == self.colour:
                    action = [int(x) for x in s[1].split(",")]
                    self.board[action[0]][action[1]] = self.opp_colour()

                    self.make_move()

        return False

    def make_move(self):
        """Makes a random move from the available pool of choices. If it can
        swap, chooses to do so 50% of the time.
        """

        # print(f"{self.colour} making move")
        # if self.colour == "B" and self.turn_count == 0:
        #     if choice([0, 1]) == 1:
        #         self.s.sendall(bytes("SWAP\n", "utf-8"))
        #     else:
        #         # same as below
        #         choices = []
        #         this_state = [0]*25
        #         for i in range(self.board_size):
        #             for j in range(self.board_size):
        #                 if self.board[i][j] == 0:
        #                     choices.append((i, j))
        #         pos = choice(choices)
        #         self.s.sendall(bytes(f"{pos[0]},{pos[1]}\n", "utf-8"))
        #         self.board[pos[0]][pos[1]] = self.colour
        # else:
        choices = []
        this_state = [0]*(BOARD_SIZE*BOARD_SIZE)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    this_state[i*BOARD_SIZE + j] = 0
                    choices.append(i*BOARD_SIZE + j)
                elif self.board[i][j] == "R":
                    this_state[i*BOARD_SIZE + j] = 1
                elif self.board[i][j] == "B":
                    this_state[i*BOARD_SIZE + j] = -1
        this_state = np.reshape(this_state, (1,) + (BOARD_SIZE, BOARD_SIZE, 1))
        action_probs, leaf_value = self.test_Model.predict(this_state, verbose=0)

        combined_data = [(action_probs[0][i], i) for i in choices]
        sorted_data = sorted(combined_data, key=lambda x: x[0], reverse=True)
        pos_index = sorted_data[0][1]

        pos = [pos_index//BOARD_SIZE, pos_index%BOARD_SIZE]

        self.s.sendall(bytes(f"{pos[0]},{pos[1]}\n", "utf-8"))
        self.board[pos[0]][pos[1]] = self.colour
        self.turn_count += 1

    def opp_colour(self):
        """Returns the char representation of the colour opposite to the
        current one.
        """
        if self.colour == "R":
            return "B"
        elif self.colour == "B":
            return "R"
        else:
            return "None"


if (__name__ == "__main__"):
    agent = NaiveAgent()
    agent.run()
