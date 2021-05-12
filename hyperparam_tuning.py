import argparse

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0'], default='CartPole-v0')
parser.add_argument('--evaluate_freq', type=int, default=50, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

def train_agent(eps_start, eps_end, anneal_length):
    
    CartPoleConfigs = {
    'memory_size': 50000,
    'n_episodes': 1000,
    'batch_size': 32,
    'target_update_frequency': 100,
    'train_frequency': 1,
    'gamma': 0.95,
    'lr': 1e-4,
    'n_actions': 2,
    }
    
    CartPoleConfigs['eps_start'] = eps_start
    CartPoleConfigs['eps_end'] = eps_end
    CartPoleConfigs['anneal_length'] = anneal_length
    
    
    print(f"device: {device}")
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make('CartPole-v0')
    env_config = CartPoleConfigs
    
    print(env_config)

    print('----')
    print('constructing DQN')
    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # Create target network and set parameters to the same as dqn.
    target = DQN(env_config=env_config).to(device)
    target.load_state_dict(dqn.state_dict())

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])
    
    # Training phase values
    training_phase_values = list()
    
    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    
    # ! step counter has to be outside to work properly
    step = 0
    for episode in range(201):#env_config['n_episodes']):
        done = False

        obs = preprocess(env.reset(), env=args.env).unsqueeze(0)
        
        while not done:
            step += 1
            
            # ! torch no grad should be better here I think!
            # saves some computing power #optimize
            with torch.no_grad():
                action = dqn.act(obs)
                # Act in the true environment.
                obs_next, reward, done, info = env.step(action.item())

                # Preprocess incoming observation.
                # ! always preprocess no matter what.
                obs_next = preprocess(obs_next, env=args.env).unsqueeze(0)
                
                # Add the transition to the replay memory. 
                # TODO: Remember to convert everything to PyTorch tensors!
                
                memory.push(obs, action, obs_next, torch.tensor(reward), torch.tensor((not done)))
                obs = obs_next
                
                

            # Run DQN.optimize() every env_config["train_frequency"] steps.
            if step % env_config["train_frequency"] == 0:
                optimize(dqn, target, memory, optimizer)

            # Update the target network every env_config["target_update_frequency"] steps.
            if step % env_config["target_update_frequency"] == 0:
                target.load_state_dict(dqn.state_dict())

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            
            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')
            training_phase_values.append((episode, mean_return))
            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far!') # Saving model.')
                #torch.save(dqn, f'models/{args.env}_best.pt')
        
    # Close environment after training is completed.
    env.close()
    
    return np.array(training_phase_values)
    
    
    
if __name__ == "__main__":
    
    # Annealing_length tuning
    # We fix eps_start=0.9 & eps_end=0.1
    
    for annealing_exp_i in [1000,500,250,100,1]:
        print("#######################################################")
        print(f'Annealing_Experiment: anneal_length={annealing_exp_i}')
        print("#######################################################")
        for replicate_j in range(3):
            print(f'\treplicate : {replicate_j}') 
            x = train_agent(eps_start=1, eps_end=0.05, anneal_length=annealing_exp_i)
            print(x)
        print('')
    
    plt.show()
    
    