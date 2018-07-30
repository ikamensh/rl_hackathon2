from tic_tac_toe import TicTacToe
import numpy as np

def test_victory_condition_rows():
    game = TicTacToe()

    assert game.evaluate(game.board) is None

    game.board[0] = 1
    assert game.evaluate(game.board) == 1

    game.board[0] = -1
    assert game.evaluate(game.board) == -1


def test_victory_condition_columns():
    game = TicTacToe()

    assert game.evaluate(game.board) is None

    game.board[:,0] = 1
    assert game.evaluate(game.board) == 1

    game.board[:,0] = -1
    assert game.evaluate(game.board) == -1

def test_victory_condition_diags():
    game = TicTacToe()

    assert game.evaluate(game.board) is None

    rows = [0,1,2]
    columns = [2,1,0]

    game.board[rows, rows] = 1
    assert game.evaluate(game.board) == 1

    game.board[rows, columns] = -1
    assert game.evaluate(game.board) == -1


def test_draw_detected():
    game = TicTacToe()
    game.board = np.array([[ 1, -1, -1],
                           [-1,  1,  1],
                           [ 1, -1, -1],])

    assert game.evaluate(game.board) == 0

    game.board = np.array([[ 1, -1, 1],
                           [-1,  0, 1],
                           [ 1, -1,-1], ])

    assert game.evaluate(game.board) is None
