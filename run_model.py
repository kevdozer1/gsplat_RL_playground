#!/usr/bin/env python3
"""
Script to run a trained model in the PLAYGROUND environment
"""

import os
import sys
import argparse
import time
import numpy as np
from stable_baselines3 import PPO, SAC

# Add current directory to path for imports
sys.path.append(os.path.abspath('.'))

# Import environment
from playground_rl.environments.simple_robot_env import SimpleRobotEnv
import gymnasium as gym

# Create a wrapper to adapt the observations to match what the model expects
class ObservationAdapter(gym.Wrapper):
    def __init__(self, env, adaptation_mode='new_to_old'):
        super().__init__(env)
        self.adaptation_mode = adaptation_mode
        
        if adaptation_mode == 'new_to_old':
            # New environment (19-dim) to old model (15-dim)
            # Remove indices 10-13 (object orientation)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
            )
        elif adaptation_mode == 'old_to_new':
            # Old model (15-dim) to new environment (19-dim)
            # Add 4 zeros for object orientation (indices 10-13)
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
            )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        if self.adaptation_mode == 'new_to_old':
            # Remove object orientation (indices 10-13)
            adapted_obs = np.concatenate([obs[:10], obs[14:]])
            return adapted_obs, info
        elif self.adaptation_mode == 'old_to_new':
            # Add placeholder values for object orientation
            adapted_obs = np.concatenate([obs[:10], np.zeros(4), obs[10:]])
            return adapted_obs, info
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.adaptation_mode == 'new_to_old':
            # Remove object orientation (indices 10-13)
            adapted_obs = np.concatenate([obs[:10], obs[14:]])
            return adapted_obs, reward, terminated, truncated, info
        elif self.adaptation_mode == 'old_to_new':
            # Add placeholder values for object orientation
            adapted_obs = np.concatenate([obs[:10], np.zeros(4), obs[10:]])
            return adapted_obs, reward, terminated, truncated, info
        
        return obs, reward, terminated, truncated, info

def parse_args():
    parser = argparse.ArgumentParser(description='Run a trained RL agent')
    parser.add_argument('--model', type=str, default='results/models/final_model_PPO.zip',
                       help='Path to trained model file (.zip)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'SAC'],
                       help='Algorithm used for the model (PPO or SAC)')
    return parser.parse_args()

def get_object_shape_name(env):
    """Get the name of the currently selected object shape"""
    if not hasattr(env.env, 'selected_object'):  # Access through wrapper
        return "Unknown"
    
    obj_type = env.env.selected_object.get('type', 'Unknown')
    return obj_type.capitalize()

def main():
    args = parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return
    
    # Load the trained model with the appropriate algorithm
    print(f"Loading {args.algo} model from: {args.model}")
    if args.algo == 'PPO':
        model = PPO.load(args.model)
    else:
        model = SAC.load(args.model)
    
    # Create environment with visualization
    raw_env = SimpleRobotEnv(render_mode="human" if not args.no_render else None)
    
    # Determine the adaptation mode based on the model
    if "multiple_shapes" in args.model or "ppo_multiple_shapes" in args.model:
        # Model trained with newer environment - already expects 15-dim observations
        env = ObservationAdapter(raw_env, adaptation_mode='new_to_old')
        print("Using new-to-old observation adapter for multiple shapes model")
    else:
        # Old model - expects 15-dim observations from older environment
        env = ObservationAdapter(raw_env, adaptation_mode='new_to_old')
        print("Using new-to-old observation adapter for standard model")
    
    # Run episodes
    total_rewards = []
    shape_stats = {"Box": 0, "Sphere": 0, "Cylinder": 0, "Unknown": 0}
    
    for episode in range(args.episodes):
        observation, info = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        # Track the shape being used in this episode
        current_shape = get_object_shape_name(env)
        shape_stats[current_shape] += 1
        
        print(f"Starting episode {episode+1}/{args.episodes} with shape: {current_shape}")
        
        while not done:
            # Get action from the trained model
            action, _ = model.predict(observation, deterministic=True)
            
            # Execute the action
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Check if episode is done
            done = terminated or truncated
            
            # Slow down for visualization
            if not args.no_render:
                time.sleep(0.01)
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step}, Shape = {current_shape}")
    
    # Print summary
    print("\nResults Summary:")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Min reward: {np.min(total_rewards):.2f}")
    print(f"Max reward: {np.max(total_rewards):.2f}")
    
    # Print shape distribution
    print("\nShape Distribution:")
    for shape, count in shape_stats.items():
        if count > 0:
            print(f"{shape}: {count} episodes")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()