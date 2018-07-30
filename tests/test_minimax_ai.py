from tic_tac_toe import TicTacToe
from minimax_ai import MinimaxAI
import numpy as np

def test_makes_valid_turns():
    game = TicTacToe()
    ai = MinimaxAI(game)

    row, column = ai.decide_turn()

    assert 0 <= row <= 2
    assert 0 <= column <= 2


def test_terminates():
    game = TicTacToe()
    game.board[:] = 1
    game.board[0, 0] = 0

    ai = MinimaxAI(game)

    row, column = ai.decide_turn()



def test_chooses_winning_move():
    game = TicTacToe()
    game.x_next = True
    game.board = np.array([ [   -1,  1,  0],
                            [    1,  1, -1],
                            [   -1,  0,  1]  ])

    ai = MinimaxAI(game)

    row, column = ai.decide_turn()

    assert (row, column) == (2,1)


def test_chooses_blocking_move():
    game = TicTacToe()
    game.x_next = False
    game.board = np.array([ [   -1,  1,  0],
                            [    1,  1, -1],
                            [   -1,  0,  1]  ])

    ai = MinimaxAI(game)

    row, column = ai.decide_turn()

    assert (row, column) == (2,1)