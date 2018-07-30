import numpy as np
import random
from functools import lru_cache
from gym.core import Env
from ai.minimax_ai import MinimaxAI
from gym import spaces



class TicTacToe(Env):

    def __init__(self):
        self.board = np.zeros([3,3], dtype=np.int8)
        self.x_next = True if random.random() > 0.5 else False

        self.x_ai = None
        self.o_ai = MinimaxAI(self)
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1, high=1, dtype=np.int8, shape=(3,3))


    def reset(self):
        self.board = np.zeros([3, 3])
        self.x_next = True if random.random() > 0.5 else False
        if not self.x_next:
            row, column = self.o_ai.decide_turn()
            assert self.try_make_turn(row, column)

        return self.board

    def step(self, action):
        """
        TODO this method is not complete. its up to you to finish it.
        :param action: a discrete action from 0 to 8. Cells are enumerated from left to right,
        with columns continuing from top to bottom, like below:

         0 | 1 | 2
        -  + - + -
         3 | 4 | 5
        -  + - + -
         6 | 7 | 8

        if invalid action is used, board does not change, but negative reward is returned.
        :return: observation, reward, done?, info (empty)
        """
        assert self.x_next # verify it is your turn

        # extract row and column from your actions

        row = action // 3
        column = action % 3

        # Info is an empty dictionary. a dictionary is required by the keras-rl specs.
        info = {}

        # use the row and column to try to make a step

        # if the turn is not a valid one, the game ends and we lose it.
        # The reward should be negative and greater than a game lost normally.

        # check it your turn has ended the game.
        # if not, opponent should make his turn, and we check again if the game is over.

        #if the game is over, we return the observation, reward = self.evaluate(self.board), done = True and the info.

        # if the game is not over, we return the observation, reward = 0, done = False and the info.

        # TODO default return - remove it when you implement a real one.
        return np.zeros([3,3]), 0, False, info


    def try_make_turn(self, row, column):

        if self.board[row,column] == 0:

            self.board[row, column] = 1 if self.x_next else -1
            self.x_next = not self.x_next
            return True

        else:
            return False

    @staticmethod
    def evaluate(board):
        board_as_tuple = tuple(tuple(board[row]) for row in range(3))
        return TicTacToe._evaluate(board_as_tuple)

    def play_a_game(self):
        while self.evaluate(self.board) is None:
            ai = self.x_ai if self.x_next else self.o_ai
            row, column = ai.decide_turn()
            valid_turn = self.try_make_turn(row, column)
            if not valid_turn:
                # any AI that makes invalid turn immediately loses the game.
                return -1 if self.x_next else 1
        return self.evaluate(self.board)

    @property
    def board_as_tuple(self):
        return tuple(tuple(self.board[row]) for row in range(3))


    @staticmethod
    @lru_cache(maxsize=2**16)
    def _evaluate(board):

        sums = []
        # we collect totals of all row, columns and diagonals. Any of those must have a value of
        # either 3 or -3 if there are x x x or o o o in this sequence.
        sums += [sum(board[row]) for row in range(3)]
        sums += [sum([board[i][column] for i in range(3)]) for column in range(3)]
        sum_main_diag = sum([board[i][i] for i in range(3)])
        sum_opp_diag = sum([board[i][2 - i] for i in range(3)])

        sums.append(sum_main_diag)
        sums.append(sum_opp_diag)

        if 3 in sums:
            return 1
        elif -3 in sums:
            return -1
        else:
            n_empty = sum([1 for row in range(3) for cell in board[row] if cell == 0])
            if n_empty == 0:
                return 0
            else:
                return None


    def render(self, mode="Human"):
        shapes = {-1: 'o', 0: ' ', 1: 'x'}
        print(f"{shapes[self.board[0,0]]}|{shapes[self.board[0,1]]}|{shapes[self.board[0,2]]}")
        print('-+-+-')
        print(f"{shapes[self.board[1,0]]}|{shapes[self.board[1,1]]}|{shapes[self.board[1,2]]}")
        print('-+-+-')
        print(f"{shapes[self.board[2,0]]}|{shapes[self.board[2,1]]}|{shapes[self.board[2,2]]}")













































