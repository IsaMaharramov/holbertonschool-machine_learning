#!/usr/bin/env python3
"""
SARSA(λ) Value Estimation for Reinforcement Learning.
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    This function performs the SARSA(λ) algorithm to estimate the Q table.
    Args:
        - env: The environment instance.
        - Q: numpy.ndarray of shape (s, a) containing the Q table.
        - lambtha: The eligibility trace factor.
        - episodes: Number of episodes to train over.
        - max_steps: Maximum number of steps per episode.
        - alpha: Learning rate.
        - gamma: Discount rate.
        - epsilon: Initial threshold for epsilon greedy.
        - min_epsilon: Minimum value that epsilon should decay to.
        - epsilon_decay: Decay rate for updating epsilon between episodes.
    Returns:
        - Updated Q, the Q table.
    """
    # Get the value of epsilon for the first episode
    initial_epsilon = epsilon

    # Iterate over each episode
    for episode in range(episodes):
        # Initialize eligibility traces for all state-action pairs to 0
        E = np.zeros_like(Q)
        # Reset environment and get the initial state
        state = env.reset()[0]

        # Choose the first action using epsilon-greedy policy
        if np.random.uniform() < epsilon:
            action = np.random.randint(Q.shape[1])
        else:
            action = np.argmax(Q[state])

        # Iterate over steps in the episode
        for _ in range(max_steps):
            # Take the chosen action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Choose the next action using epsilon-greedy policy from
            # next_state
            if np.random.uniform() < epsilon:
                next_action = np.random.randint(Q.shape[1])
            else:
                next_action = np.argmax(Q[next_state])

            # Compute the TD error (delta)
            delta = (reward + gamma * Q[next_state, next_action] -
                     Q[state, action])

            # Update the eligibility trace for the current state-action pair
            E[state, action] += 1

            # Update the Q table for all state-action pairs using the
            # eligibility
            # traces
            Q += alpha * delta * E

            # Decay eligibility traces for all state-action pairs
            E *= gamma * lambtha

            # Update state and action for the next step
            state, action = next_state, next_action

            # If the episode is finished, break out of the loop
            if terminated or truncated:
                break

        # Exploration rate decay
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))
    return Q
