#!/usr/bin/env python3
"""
Train an agent to play Atari's Breakout using Deep Q-learning.
"""
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


class KerasRLWrapper(gym.Wrapper):
    """
    Gymnasium wrapper to ensure compatibility with keras-rl2.
    """
    def __init__(self, env):
        """ Initialize the wrapper """
        super().__init__(env)

    def step(self, action):
        """ Modify step to return 4 values for keras-rl2 """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def reset(self, **kwargs):
        """ Modify reset to return 1 value for keras-rl2 """
        obs, info = self.env.reset(**kwargs)
        return obs

    def render(self, *args, **kwargs):
        """ Safely render the environment ignoring unexpected kwargs """
        return self.env.render()


def build_model(window_length, num_actions):
    """ Build the neural network model for DQN """
    model = Sequential()
    # Permute to fit Keras expectation of channels-last (84, 84, 4)
    model.add(Permute((2, 3, 1), input_shape=(window_length, 84, 84)))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    return model


if __name__ == '__main__':
    # Initialize environment
    env = gym.make("ALE/Breakout-v5")
    # Preprocess handles grayscaling, resizing to 84x84, and frame skipping
    env = AtariPreprocessing(env, frame_skip=4, terminal_on_life_loss=True,
                             scale=True)
    env = KerasRLWrapper(env)

    window_length = 4
    num_actions = env.action_space.n

    # Build model, memory, and policy
    model = build_model(window_length, num_actions)
    memory = SequentialMemory(limit=1000000, window_length=window_length)
    policy = EpsGreedyQPolicy(eps=0.1)

    # Configure the DQN agent
    dqn = DQNAgent(model=model, nb_actions=num_actions, policy=policy,
                   memory=memory, nb_steps_warmup=1000,
                   target_model_update=1000)

    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    
    # Train agent (Set nb_steps higher like 1000000 for full convergence)
    dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)
    
    # Save the trained weights
    dqn.save_weights('policy.h5', overwrite=True)
