#!/usr/bin/env python3
"""
Module for computing a simple policy function for reinforcement learning.
"""
import numpy as np


def policy(matrix, weight):
    """
    This function computes the policy given a state matrix and a weight matrix.
    It applies a linear transformation on the state using the weight
    matrix and then applies the softmax function to obtain a probability
    distribution over actions.
    Args:
        matrix (np.ndarray): A 2D numpy array representing the state.
        weight (np.ndarray): A 2D numpy array representing the weight matrix.
    Returns:
        np.ndarray: A 2D numpy array containing the policy probabilities for
            each action.
    """
    # Compute the linear combination of the state and the weights
    z = np.dot(matrix, weight)
    # Apply the softmax function to obtain probabilities
    exp_z = np.exp(z)
    softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return softmax


def policy_gradient(state, weight):
    """
    This function computes the Monte-Carlo policy gradient for a given state
    and weight matrix.
    It first computes the action probabilities using the policy function,
    then samples an action based on these probabilities, and finally
    calculates the gradient of the log probability of the chosen action with
    respect to the weight matrix.
    Args:
        state (np.ndarray): A 1D or 2D numpy array representing the current
            observation (state).
        weight (np.ndarray): A 2D numpy array representing the weight matrix.
    Returns:
        int: The sampled action.
        np.ndarray: The gradient of the log probability of the chosen action
            with respect to the weight.
    """
    # Reshape state to 2D if it is a 1D array (to ensure proper matrix
    # multiplication)
    if state.ndim == 1:
        state = state.reshape(1, -1)

    # Compute action probabilities using the policy function
    probs = policy(state, weight)  # Shape: (1, num_actions)

    # Sample an action using the probability distribution (flattening
    # probs to 1D)
    action = np.random.choice(probs.shape[1], p=probs.flatten())

    # Create a one-hot vector for the chosen action (shape: (1, num_actions))
    one_hot = np.zeros_like(probs)
    one_hot[0, action] = 1

    # Compute the gradient: state.T dot (one_hot - probs)
    # This yields a gradient of shape (num_features, num_actions)
    grad = np.dot(state.T, (one_hot - probs))

    return action, grad
