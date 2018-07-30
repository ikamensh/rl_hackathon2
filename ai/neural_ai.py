from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent


WINDOW_LENGTH = 4
input_shape = (3, 3)
input_shape = (WINDOW_LENGTH,) + input_shape
nb_actions = 9


class NeuralAI:

    def __init__(self, game, weights_path=None):
        self.game = game
        model = self.build_model()
        dqn = DQNAgent(model=model, nb_actions=nb_actions, gamma=.99,
                       memory=SequentialMemory(limit=10, window_length=WINDOW_LENGTH))

        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        weights_filename = weights_path or 'saved_dqn_TicTacToe_weights.h5f'
        dqn.load_weights(weights_filename)
        self.dqn = dqn

    def decide_turn(self):
        board = self.game.board
        action = self.dqn.forward(board)
        row = action // 3
        column = action % 3

        return row, column

    @staticmethod
    def build_model():
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(nb_actions))
        return model



