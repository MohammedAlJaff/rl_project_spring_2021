import argparse
import random

import gym
import torch
import torch.nn as nn

import config
from utils import preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0'])
parser.add_argument('--path', type=str, help='Path to stored DQN model.')
parser.add_argument('--n_eval_episodes', type=int, default=1, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--render', dest='render', action='store_true', help='Render the environment.')
parser.add_argument('--save_video', dest='save_video', action='store_true', help='Save the episodes as video.')
parser.set_defaults(render=False)
parser.set_defaults(save_video=False)

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole,
    'Pong-v0': config.Pong,
}

def evaluate_policy(dqn, env, env_config, args, n_episodes, render=False, verbose=False):
    """Runs {n_episodes} episodes to evaluate current policy."""
    total_return = 0

    for i in range(n_episodes):
        obs = torch.tensor(env.reset()).float()
        
        obs_stack = torch.stack(env_config['observation_stack_size'] * [obs]).unsqueeze(0).to(device)

        done = False
        episode_return = 0

        while not done:
            if render:
                env.render()
            
            action = dqn.map_action(dqn.act(obs_stack, exploit=True))

            next_obs, reward, done, info = env.step(action)
            next_obs = torch.tensor(next_obs).float().unsqueeze(0)
            
            obs_stack = torch.cat((obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)

            episode_return += reward
        
        total_return += episode_return
        
        if verbose:
            print(f'Finished episode {i+1} with a total return of {episode_return}')

    
    return total_return / n_episodes

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config
    env = gym.make(args.env)
    env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30, scale_obs=True)
    
    env_config = ENV_CONFIGS[args.env]

    if args.save_video:
        env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: True, force=True)

    # Load model from provided path.
    dqn = torch.load(args.path, map_location=torch.device('cpu'))
    dqn.eval()

    mean_return = evaluate_policy(dqn, env, env_config, args, args.n_eval_episodes, render=args.render and not args.save_video, verbose=True)
    print(f'The policy got a mean return of {mean_return} over {args.n_eval_episodes} episodes.')

    env.close()