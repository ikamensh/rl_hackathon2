from __future__ import division
import argparse


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from tic_tac_toe import TicTacToe



parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='TicTacToe')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()


from gym.wrappers import TimeLimit
env = TimeLimit(TicTacToe(), max_episode_steps=10)
nb_actions = env.action_space.n


WINDOW_LENGTH = 4
input_shape = (3,3)
input_shape = (WINDOW_LENGTH,) + input_shape

def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(nb_actions))
    print(model.summary())
    return model


model = build_model()


memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)


dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, nb_steps_warmup=50000, gamma=.99,
               target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

print("about to enter the flags branching.")

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    print("about to fit the dqn!")
    dqn.load_weights(weights_filename)
    dqn.fit(env, callbacks=callbacks, nb_steps=int(1e7), log_interval=10000, verbose=1)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)

elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)