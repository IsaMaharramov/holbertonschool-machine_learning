#!/usr/bin/env python3
"""
Play Atari's Breakout using a trained agent.
"""
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy


class KerasRLWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for keras-rl compatibility.
    """
    def __init__(self, env):
        """ Initialize the wrapper """
        super().__init__(env)

    def step(self, action):
        """ Modify step to return 4 values """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def reset(self, **kwargs):
        """ Modify reset to return 1 value """
        obs, info = self.env.reset(**kwargs)
        return obs

    def render(self, *args, **kwargs):
        """ Render the environment safely """
        return self.env.render()


def build_model(window_length, num_actions):
    """ Build the neural network model for DQN """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(window_length, 84, 84)))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    return model


if __name__ == '__main__':
    # Initialize environment with render mode set to 'human' for display
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = AtariPreprocessing(env, frame_skip=4, terminal_on_life_loss=False,
                             scale=True)
    env = KerasRLWrapper(env)

    window_length = 4
    num_actions = env.action_space.n

    # Set up same model architecture
    model = build_model(window_length, num_actions)
    memory = SequentialMemory(limit=1000000, window_length=window_length)
    
    # Requirement: Utilize GreedyQPolicy during gameplay
    policy = GreedyQPolicy()

    # Configure the DQN agent
    dqn = DQNAgent(model=model, nb_actions=num_actions, policy=policy,
                   memory=memory, nb_steps_warmup=1000,
                   target_model_update=1000)

    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    
    # Load previously trained weights
    dqn.load_weights('policy.h5')
    
    # Let the agent play the game
    dqn.test(env, nb_episodes=5, visualize=True)
