"""
In this file, you may edit the hyperparameters used for different environments.

    1. memory_size: Maximum size of the replay memory.

    2. n_episodes: Number of episodes to train for.

    3. batch_size: Batch size used for training DQN.

    4. target_update_frequency: How often to update the target network.

    5. train_frequency: How often to train the DQN.

    6. gamma: Discount factor.

    7. lr: Learning rate used for optimizer.

    8. eps_start: Starting value for epsilon (linear annealing).

    9. eps_end: Final value for epsilon (linear annealing).

    10. anneal_length: How many steps to anneal epsilon for.

    11. n_actions: The number of actions can easily be accessed with env.action_space.n, but we do
                    some manual engineering to account for the fact that Pong has duplicate actions.

"""

# Hyperparameters for CartPole-v0
CartPole = {
    'memory_size': 50000,
    'n_episodes': 1000,
    'batch_size': 32,
    'target_update_frequency': 100,
    'train_frequency': 1,
    'gamma': 0.95,
    'lr': 1e-4,
    'eps_start': 1.0,
    'eps_end': 0.05,
    'anneal_length': 10**4,
    'n_actions': 2,
}

# majnotes - Hyperparameters for Pong. 
Pong = {
    'memory_size': 10000,
    'n_episodes': 2000,
    'batch_size': 32,
    'target_update_frequency': 1000,
    'train_frequency': 4,
    'gamma': 0.99,
    'lr': 1e-4,
    'eps_start': 1.0,
    'eps_end': 0.01,
    'anneal_length': 10**6,
    'n_actions': 2,
    'observation_stack_size': 4,
}