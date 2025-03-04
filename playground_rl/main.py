#!/usr/bin/env python3
"""
Main entry point for PLAYGROUND project.
This script provides a command-line interface to run different components
of the PLAYGROUND robotic manipulation system.
"""

import os
import sys
import argparse
import yaml
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Import local modules
from playground_rl.environments.simple_robot_env import SimpleRobotEnv
from playground_rl.utils.mesh_utils import (
    load_mesh, visualize_mesh, mesh_to_vhacd, load_mesh_pybullet
)

# Import the training script
from playground_rl.train import main as train_main


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


def test_environment(render=True, steps=1000):
    """Test the environment with random actions."""
    print("Testing environment with random actions...")
    env = SimpleRobotEnv(render_mode="human" if render else None)
    observation, info = env.reset()
    
    rewards = []
    for i in range(steps):
        # Random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        if terminated or truncated:
            observation, info = env.reset()
            print(f"Episode terminated after {i+1} steps")
    
    env.close()
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Rewards over steps')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('random_rewards.png')
    plt.show()


def run_trained_agent(model_path, render=True, episodes=5):
    """Run a trained agent in the environment."""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    env = SimpleRobotEnv(render_mode="human" if render else None)
    
    for episode in range(episodes):
        observation, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated
            
            if render:
                time.sleep(0.01)  # Slow down for visualization
        
        print(f"Episode {episode+1}: Reward = {episode_reward}, Steps = {step}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description='PLAYGROUND: Robotic manipulation with RL')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train an RL agent')
    train_parser.add_argument('--config', type=str, default='configs/default.yaml',
                             help='Path to config file')
    train_parser.add_argument('--render', action='store_true',
                             help='Render training')
    
    # Test environment command
    test_env_parser = subparsers.add_parser('test-env', help='Test environment with random actions')
    test_env_parser.add_argument('--steps', type=int, default=1000,
                               help='Number of steps')
    test_env_parser.add_argument('--no-render', action='store_true',
                                help='Disable rendering')
    
    # Run trained agent command
    run_parser = subparsers.add_parser('run', help='Run a trained agent')
    run_parser.add_argument('--model', type=str, required=True,
                           help='Path to trained model')
    run_parser.add_argument('--episodes', type=int, default=5,
                           help='Number of episodes')
    run_parser.add_argument('--no-render', action='store_true',
                           help='Disable rendering')
    
    # Visualize mesh command
    viz_parser = subparsers.add_parser('viz-mesh', help='Visualize a mesh file')
    viz_parser.add_argument('--mesh', type=str, required=True,
                          help='Path to mesh file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Simply pass the arguments to the train_main function
        # Remove the config parameter since it's not accepted by train_main
        train_main()
    
    elif args.command == 'test-env':
        test_environment(render=not args.no_render, steps=args.steps)
    
    elif args.command == 'run':
        run_trained_agent(args.model, render=not args.no_render, episodes=args.episodes)
    
    elif args.command == 'viz-mesh':
        mesh = load_mesh(args.mesh)
        if mesh:
            visualize_mesh(mesh, title=f"Mesh: {os.path.basename(args.mesh)}")
        else:
            print(f"Failed to load mesh: {args.mesh}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()