#!/usr/bin/env python3
"""
Module for training a policy gradient agent using the REINFORCE algorithm.
"""
import numpy as np
import gymnasium as gym
# Import the policy_gradient function from the policy_gradient module
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    This funtion trains a policy gradient agent using Monte-Carlo updates
    (REINFORCE).
    Args:
        env: The initial environment.
        nb_episodes (int): Number of episodes to use for training.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
    Returns:
        list: A list of scores (sum of rewards) for each episode.
    """
    # Initialize the weight matrix for the policy.
    # For CartPole-v1, the state has 4 features and there are 2 possible
    # actions.
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)

    scores = []  # List to store the score for each episode

    # Loop over the specified number of episodes
    for episode in range(nb_episodes):
        # Reset the environment and get the initial observation
        state, _ = env.reset()
        episode_rewards = []  # To store rewards for the current episode
        grads = []          # To store the gradient for each time step

        # Initialize done and truncated flags (gymnasium returns both)
        done = False
        truncated = False

        # Run the episode until it is finished
        while not (done or truncated):
            # Get an action and the gradient for the current state using the
            # policy_gradient function
            action, grad = policy_gradient(state, weight)
            action = int(action)  # Ensure action is an integer

            # Append the gradient from this step
            grads.append(grad)

            # Take the action in the environment and obtain the next state and
            # reward
            next_state, reward, done, truncated, _ = env.step(action)

            # Append the reward for this step
            episode_rewards.append(reward)

            # Update the state to the new observation
            state = next_state

        # Compute the discounted returns for the episode.
        returns = []
        G = 0
        # Loop backwards over the rewards to compute the return at each step
        for r in reversed(episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = np.array(returns)

        # Update the policy weights using the computed gradients and returns.
        # The weight update is done using gradient ascent:
        # weight = weight + alpha * (return at step t) * (gradient at step t)
        for i in range(len(grads)):
            weight += alpha * returns[i] * grads[i]

        # Compute the total reward (score) for this episode
        total_reward = sum(episode_rewards)
        scores.append(total_reward)

        # Print the current episode and score in the required format
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Score {total_reward}")

        # Render every 1000 episodes if show_result is True
        if show_result and (episode + 1) % 1000 == 0:
            env.render()

    return scores
