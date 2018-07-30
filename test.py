from tic_tac_toe import TicTacToe
from ai.random_ai import RandomAI
from ai.minimax_ai import MinimaxAI
from ai.neural_ai import NeuralAI



ttt = TicTacToe()

x_ai = NeuralAI(ttt)
ttt.x_ai = x_ai

o_ai = MinimaxAI(ttt)
ttt.o_ai = o_ai


counters = {-1:0, 0:0, 1:0}
n_games = int(1e5)

def run_trials():
    for i in range(n_games):
        if i % 50 == 0:
            print(i, "out of", n_games)
            print(f"stats: out of {n_games} games, x has won {counters[1]} times, o - {counters[-1]} times, and there were "
                  f"{counters[0]} draws.")

        ttt.reset()
        result = ttt.play_ai_game()
        counters[result] += 1



run_trials()



# from cProfile import Profile
# profiler = Profile()
# profiler.runcall(run_trials)
# profiler.print_stats('cumulative')




