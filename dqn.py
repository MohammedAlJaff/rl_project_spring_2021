import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.eps = self.eps_start
        self.eps_delta = (self.eps_start - self.eps_end) / self.anneal_length  # ToDo
        self.n_actions = env_config["n_actions"]

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.
        n_observations = observation.shape[0]
        actions = torch.zeros((n_observations, 1), dtype=torch.int)
        random_numbers = np.random.uniform(size=n_observations)
        action_values = self.forward(observation)
        for i in range(n_observations):
            if random_numbers[i] <= self.eps and not exploit:
                # Explore
                actions[i, 0] = np.random.choice(self.n_actions)
            else:
                # Exploit. If several actions are optimal, choose randomly among them
                best_actions = torch.where(action_values[i] == torch.max(action_values[i]))[0]
                actions[i, 0] = random.choice(best_actions).item()
            # Update exploration rate
            if self.eps > self.eps_end:
                self.eps -= self.eps_delta
        return actions


def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!
    observations, actions, next_observations, rewards = memory.sample(target_dqn.batch_size)

    observations = torch.stack(observations)
    n_obs, _, n_actions = observations.shape
    observations = observations.reshape(n_obs, n_actions).to(device)

    actions = torch.stack(actions).reshape(n_obs, 1).to(device)

    next_observations, terminal_indices = handle_terminal_transitions(next_observations, n_actions)
    next_observations = torch.stack(next_observations).reshape(n_obs, n_actions).to(device)

    rewards = torch.stack(rewards).to(device)

    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    q_values = dqn.forward(observations).gather(1, actions)

    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
    v = torch.ones(1, n_obs)
    v[0, terminal_indices] = 0
    q_value_targets = rewards + target_dqn.gamma * v * target_dqn.forward(next_observations).max(dim=1).values
    # ToDo: What to do with non-terminal transitions?

    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()


def handle_terminal_transitions(next_obs, n_actions):
    """
    Takes the next observations from a memory sample, finds the indices of the terminating states and returns them
    as a list together with am updated observations sample
    """
    terminal_indices = [i for i in range(len(next_obs)) if isinstance(next_obs[i], np.ndarray)]
    next_observations = list(next_obs)
    for i in terminal_indices:
        next_observations[i] = torch.zeros(1, n_actions)
    return tuple(next_observations), terminal_indices
