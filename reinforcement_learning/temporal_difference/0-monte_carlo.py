#!/usr/bin/env python3
"""
Monte Carlo Value Estimation for Reinforcement Learning.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    This function performs the Monte Carlo algorithm to estimate the value
    function. It updates the value estimate V using the incremental mean.
    Args:
    - env: The environment instance.
    - V: numpy.ndarray of shape (s,) containing the value estimate.
    - policy: Function that takes a state and returns the next action.
    - episodes: Number of episodes to train over.
    - max_steps: Maximum number of steps per episode.
    - alpha: Learning rate.
    - gamma: Discount rate.
    Returns:
    - Updated V, the updated value estimate.
    """
    # Iterate over the number of episodes
    for episode in range(episodes):
        state = env.reset()[0]  # Initialize state
        episode_data = []  # Store state, reward sequence

        # Generate an episode
        for step in range(max_steps):
            action = policy(state)  # Choose action based on policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))  # Store state and reward

            if terminated or truncated:
                break  # End episode if done

            # move to the next state
            state = next_state

        # Compute returns in reverse order
        G = 0  # Initialize return
        # Convert to numpy array
        episode_data = np.array(episode_data, dtype=int)

        for state, reward in reversed(episode_data):
            G = reward + gamma * G  # Compute return

            # Update V(s) using incremental mean
            if state not in episode_data[:episode, 0]:  # First visit MC
                V[state] += alpha * (G - V[state])  # Update rule

    return V  # Return updated value function
