#!/usr/bin/env python3
"""
Training script for PLAYGROUND Phase 1
Trains a simple RL policy to interact with a cube using PyBullet and Stable Baselines3
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym

# Import our environment
from playground_rl.environments.simple_robot_env import SimpleRobotEnv

def parse_args():
    parser = argparse.ArgumentParser(description='Train an RL agent for robot manipulation')
    parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'SAC'],
                        help='RL algorithm to use (default: PPO)')
    parser.add_argument('--timesteps', type=int, default=1000000,
                        help='Total timesteps to train (default: 1000000)')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency in timesteps (default: 10000)')
    parser.add_argument('--save-freq', type=int, default=50000,
                        help='Model saving frequency in timesteps (default: 50000)')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save results (default: ./results)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num-envs', type=int, default=4,
                        help='Number of parallel environments (default: 4)')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during training')
    
    return parser.parse_args()

def make_env(render_mode=None, seed=0):
    """
    Create a function that returns an environment instance with the specified parameters.
    """
    def _init():
        env = SimpleRobotEnv(render_mode=render_mode)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, 'models')
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up random seeds
    np.random.seed(args.seed)
    
    # Create and check environment
    print("Creating and validating environment...")
    test_env = SimpleRobotEnv()
    check_env(test_env)
    
    # Create vectorized environment
    render_mode = "human" if args.render else None
    
    if args.num_envs > 1:
        # Parallel environments
        env = SubprocVecEnv([make_env(render_mode=None, seed=args.seed + i) for i in range(args.num_envs)])
        # Create a separate environment for evaluation
        eval_env = DummyVecEnv([make_env(render_mode=render_mode, seed=args.seed + args.num_envs)])
    else:
        # Single environment
        env = DummyVecEnv([make_env(render_mode=render_mode, seed=args.seed)])
        eval_env = env
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.num_envs,
        save_path=model_dir,
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, 'best'),
        log_path=log_dir,
        eval_freq=args.eval_freq // args.num_envs,
        deterministic=True,
        render=False
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    # Create and train the model
    if args.algo == 'PPO':
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=log_dir,
            seed=args.seed,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
        )
    elif args.algo == 'SAC':
        model = SAC(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            tensorboard_log=log_dir,
            seed=args.seed,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=100,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=None,
        )
    
    print(f"Training {args.algo} for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name=args.algo,
    )
    
    # Save final model
    final_model_path = os.path.join(model_dir, f"final_model_{args.algo}")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    
    # Evaluate the trained agent
    print("Evaluating trained agent...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Visualize the trained agent if requested
    if args.render:
        print("\nVisualizing trained agent performance...")
        # Create a new environment for visualization
        vis_env = SimpleRobotEnv(render_mode="human")
        obs, _ = vis_env.reset()
        
        total_reward = 0
        done = False
        
        while not done:
            # Get action from the trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute the action
            obs, reward, terminated, truncated, _ = vis_env.step(action)
            total_reward += reward
            
            # Slow down for better visualization
            time.sleep(0.01)
            
            # Check if done
            done = terminated or truncated
        
        print(f"Visualization complete. Total reward: {total_reward:.2f}")
        vis_env.close()
    
    # Close environments
    env.close()
    if env != eval_env:
        eval_env.close()

if __name__ == "__main__":
    main()