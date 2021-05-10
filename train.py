import argparse

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0', 'Pong-v0'], required=True)
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--local', type=bool, default=False)
parser.add_argument('--logwandb', type=bool, default=False)

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole,
    'Pong-v0': config.Pong,
}

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]
    
    log = args.logwandb
    
    if args.local:
        path = 'models'
    else:
        path = "drive/MyDrive/models"
    
    if log:
        # set up weights and biases
        # 1. Start a new run
        wandb.init(project='reinforcement-learning-pong')
        # 2. Save model inputs and hyperparameters
        config = wandb.config
        # save parameters in the model here!
        config.update(env_config)

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # Create target network and set parameters to the same as dqn.
    target = DQN(env_config=env_config).to(device)
    target.load_state_dict(dqn.state_dict())

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    
    # todo: (improvement) always take 4 steps of the same action (might speed up training)
    env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, scale_obs=True, frame_skip=1, noop_max=30)
    
    # ! step counter has to be outside to work properly
    step = 0
    for episode in range(env_config['n_episodes']):
        done = False
        
        obs = torch.tensor(env.reset())
        
        obs_stack = torch.stack(env_config['observation_stack_size'] * [obs]).unsqueeze(0).float().to(device)
        
        while not done:
            step += 1
            
            # ! torch no grad should be better here in order to speed up forward run
            # saves some computing power #optimize
            with torch.no_grad():
                action = dqn.act(obs_stack)
                
                # Act in the true environment.
                obs_next, reward, done, info = env.step(dqn.map_action(action))
                obs_next = torch.tensor(obs_next).float().unsqueeze(0).to(device)
                
                # Preprocess incoming observation.
                # ! always preprocess no matter what.
                # obs_next = preprocess(obs_next, env=args.env).unsqueeze(0)
                next_obs_stack = torch.cat((obs_stack[:, 1:, ...], obs_next.unsqueeze(1)), dim=1).to(device)
                
                # Add the transition to the replay memory. 
                # ! Remember to convert everything to PyTorch tensors!
                memory.push(obs_stack, action, next_obs_stack, torch.tensor(reward), torch.tensor((not done)))
                obs_stack = next_obs_stack

            # Run DQN.optimize() every env_config["train_frequency"] steps.
            if step % env_config["train_frequency"] == 0:
                optimize(dqn, target, memory, optimizer)

            # Update the target network every env_config["target_update_frequency"] steps.
            if step % env_config["target_update_frequency"] == 0:
                target.load_state_dict(dqn.state_dict())

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            
            if log:
                wandb.log({"mean return": mean_return, "episode": episode})
            
            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'{path}/{args.env}_best.pt')
            
            torch.save(dqn, f'{path}/{args.env}_current.pt')
        
    # Close environment after training is completed.
    env.close()
