from math import inf as infinity
from random_ai import RandomAI

class MinimaxAI:

    def __init__(self, game):
        self.game = game
        self.random_ai = RandomAI(game)

    def decide_turn(self):
        turns_possible = self.empty_cells()
        if len(turns_possible) == 1:
            row, column = turns_possible[0]
        elif len(turns_possible) >= 7:
            row, column = self.random_ai.decide_turn()
        else:
            chosen_node = self.minimax(10, self.game.x_next)
            row, column, _ = chosen_node
        return row, column

    def empty_cells(self):
        state = self.game.board
        results = []
        for row in range(3):
            for column in range(3):
                if state[row, column] == 0:
                    results.append((row, column))

        return results

    def minimax(self,depth, X_turn):
        state = self.game.board
        score = self.game.evaluate(state)
        game_over = score is not None

        if depth == 0 or game_over:
            return [-1, -1, score]

        if X_turn:
            best = [-1, -1, -infinity]
        else:
            best = [-1, -1, +infinity]

        for cell in self.empty_cells():
            x, y = cell[0], cell[1]
            self.game.board[x, y] = 1 if X_turn else -1
            score = self.minimax(depth - 1, not X_turn)
            self.game.board[x, y] = 0
            score[0], score[1] = x, y

            if X_turn:
                if score[2] > best[2]:
                    best = score  # max value
            else:
                if score[2] < best[2]:
                    best = score  # min value

        return best









