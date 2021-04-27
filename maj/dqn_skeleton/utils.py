import random

import gym
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v0']:
        return torch.tensor(obs, device=device).float()
    else:
        raise ValueError('Please add necessary observation preprocessing instructions to preprocess() in utils.py.')
