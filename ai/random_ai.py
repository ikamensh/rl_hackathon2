import random


class RandomAI:

    def __init__(self, game):
        self.game = game

    def decide_turn(self):
        board = self.game.board
        assert sum([1 for cell in board.flatten() if cell == 0]) > 0, "no place to make a turn!!!"

        coords = [0,1,2]
        row = random.choice(coords)
        column = random.choice(coords)

        while board[row, column] != 0:
            row = random.choice(coords)
            column = random.choice(coords)

        return row, column

