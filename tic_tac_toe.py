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

        # by design, our AI plays x and the hard-coded AI plays for o
        self.x_ai = None
        self.o_ai = MinimaxAI(self)

        # action space consists of 9 distinct actions - trying to place your shape to each of 9 cells.
        self.action_space = spaces.Discrete(9)
        # observation space is the state of the board, as numpy array with possible values of
        # -1 - cell with 'o',
        # 0  - empty cell,
        # 1  - cell with 'x'
        self.observation_space = spaces.Box(low=-1, high=1, dtype=np.int8, shape=(3,3))


    def reset(self):
        """
        Resets the board for the next game. Starting player is random.
        """
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
        assert self.x_next
        row = action // 3
        column = action % 3
        info = {}

        valid_turn = self.try_make_turn(row,column)
        if valid_turn:
            result = self.evaluate(self.board)
            if result is None:
                row, column = self.o_ai.decide_turn()
                assert self.try_make_turn(row, column)
                result = self.evaluate(self.board)

            if result is None:
                return self.board, 0, False, info
            else:
                return self.board, result, True, info

        else:
            return self.board, -1.5, False, info


    def try_make_turn(self, row, column):
        """
        The current player according to the boolean self.x_next tries to make a turn
        by placing their shape on the crossection of the crossection of :param row and :param column
        :return: True if it was a valid turn, False if such turn is not allowed (the cell is not empty).
        """

        if self.board[row,column] == 0:

            self.board[row, column] = 1 if self.x_next else -1
            self.x_next = not self.x_next
            return True

        else:
            return False

    @staticmethod
    def evaluate(board):
        """
        takes a :param board as the input ([3,3] numpy array),
        :return:
            None if the game is not over,
            1 if x won,
            -1 if o won,
            0 if its a draw
        """
        board_as_tuple = tuple(tuple(board[row]) for row in range(3))
        return TicTacToe._evaluate(board_as_tuple)

    def play_ai_game(self):
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


    def play_vs_human(self):
        print("Welcome to Tic-Tac-Toe. Your opponent is a minimax AI, who makes first turn randomly. \n"
              "Use numbers from 0 to 8 to make turns. Below is the map of numbers to cells.")
        print("""
         0 | 1 | 2
        -  + - + -
         3 | 4 | 5
        -  + - + -
         6 | 7 | 8""")
        "You play for x, your opponent plays for o"
        self.reset()
        done = False
        while not done:
            try:
                self.render()
                action = int(input(">>> "))
            except:
                print("could not parse your input. Please try again. Use numbers from 0 to 8 to make turns.")
            else:
                row = action // 3
                column = action % 3

                valid_turn = self.try_make_turn(row, column)
                if not valid_turn:
                    print("Unfortunately you can't overwrite existing shapes. \n"
                          "You have made an invalid turn and therefore lost the game.")
                    break

                result = self.evaluate(self.board)

                if result:
                    self.render()
                    if result == 1:
                        print("Congratulations! you have won the game!")
                    elif result == 0:
                        print("The game has ended in a draw.")
                    break

                row, column = self.o_ai.decide_turn()
                assert self.try_make_turn(row, column)
                result = self.evaluate(self.board)

                if result:
                    self.render()
                    if result == -1:
                        print("Minimax AI has won the game.")
                    elif result == 0:
                        print("The game has ended in a draw.")
                    break


        print("Thanks for playing Tic-Tac-Toe. Have a nice day and come back any time!")


    def render(self, mode="Human"):
        shapes = {-1: 'o', 0: ' ', 1: 'x'}
        print(f"{shapes[self.board[0,0]]} | {shapes[self.board[0,1]]} | {shapes[self.board[0,2]]}")
        print('- + - + -')
        print(f"{shapes[self.board[1,0]]} | {shapes[self.board[1,1]]} | {shapes[self.board[1,2]]}")
        print('- + - + -')
        print(f"{shapes[self.board[2,0]]} | {shapes[self.board[2,1]]} | {shapes[self.board[2,2]]}")













































