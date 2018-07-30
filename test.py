from tic_tac_toe import TicTacToe
from ai.random_ai import RandomAI
from ai.minimax_ai import MinimaxAI
from ai.neural_ai import NeuralAI

counters = {-1:0, 0:0, 1:0}
n_games = int(1e4)

ttt = TicTacToe()
x_ai = RandomAI(ttt)
o_ai = MinimaxAI(ttt)
ttt.o_ai = o_ai
ttt.x_ai = x_ai


def run_trials():
    for i in range(n_games):
        if i % 50 == 0:
            print(i, "out of", n_games)
            print(f"stats: out of {n_games} games, x has won {counters[1]} times, o - {counters[-1]} times, and there were "
                  f"{counters[0]} draws.")

        ttt.reset()
        result = ttt.play_a_game()
        counters[result] += 1

# from cProfile import Profile
#
# profiler = Profile()
# profiler.runcall(run_trials)
#
# profiler.print_stats('cumulative')

run_trials()




