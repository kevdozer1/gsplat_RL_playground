#!/usr/bin/env python3
"""
Training script for PLAYGROUND with multiple object shapes
Trains an RL policy to interact with different object shapes (box, sphere, cylinder)
"""

import os
import argparse
import numpy as np
import time
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym

# Import our environment
from playground_rl.environments.simple_robot_env import SimpleRobotEnv

# Create a wrapper to adapt the observations when continuing training from an old model
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
    parser = argparse.ArgumentParser(description='Train an RL agent for robot manipulation with multiple object shapes')
    parser.add_argument('--algo', type=str, default='PPO', choices=['PPO', 'SAC'],
                      help='RL algorithm to use (default: PPO)')
    parser.add_argument('--timesteps', type=int, default=500000,
                      help='Total timesteps to train (default: 500000)')
    parser.add_argument('--continue-training', action='store_true',
                      help='Continue training from existing model')
    parser.add_argument('--output-dir', type=str, default='./results',
                      help='Directory to save results (default: ./results)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    parser.add_argument('--num-envs', type=int, default=1,
                      help='Number of parallel environments (default: 1)')
    parser.add_argument('--render', action='store_true',
                      help='Render the environment during training')
    parser.add_argument('--cpu', action='store_true',
                      help='Force CPU use even if CUDA is available')
    return parser.parse_args()

def make_env(render_mode=None, seed=0, continue_training=False):
    """
    Create a function that returns an environment instance with the specified parameters.
    """
    def _init():
        env = SimpleRobotEnv(render_mode=render_mode)
        
        # If continuing training from an old model, use the adaptation wrapper
        if continue_training:
            # When training, we wrap the new environment to output old-style observations
            env = ObservationAdapter(env, adaptation_mode='new_to_old')
            
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

def get_object_shape_name(env):
    """Get the name of the currently selected object shape"""
    if not hasattr(env.unwrapped, 'selected_object'):
        return "Unknown"
    
    obj_type = env.unwrapped.selected_object.get('type', 'Unknown')
    return obj_type.capitalize()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, 'models')
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up device (CPU vs GPU)
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using CPU for training")
    else:
        device = torch.device("cuda")
        print(f"Using GPU for training: {torch.cuda.get_device_name(0)}")
    
    # Set up random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create and check environment
    print("Creating and validating environment...")
    test_env = SimpleRobotEnv()
    check_env(test_env)
    
    # Create vectorized environment
    render_mode = "human" if args.render else None
    
    if args.num_envs > 1:
        # Parallel environments
        env = SubprocVecEnv([make_env(render_mode=None, seed=args.seed + i, 
                                     continue_training=args.continue_training) 
                            for i in range(args.num_envs)])
        # Create a separate environment for evaluation
        eval_env = DummyVecEnv([make_env(render_mode=render_mode, seed=args.seed + args.num_envs,
                                        continue_training=args.continue_training)])
    else:
        # Single environment
        env = DummyVecEnv([make_env(render_mode=render_mode, seed=args.seed,
                                   continue_training=args.continue_training)])
        eval_env = env
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // args.num_envs, 1),
        save_path=model_dir,
        name_prefix="model_multi_shapes",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, 'best'),
        log_path=log_dir,
        eval_freq=max(20000 // args.num_envs, 1),
        deterministic=True,
        render=False
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    # Create a new model or load existing one
    if args.continue_training and os.path.exists(os.path.join(model_dir, "final_model_PPO.zip")):
        print(f"Loading existing model from {os.path.join(model_dir, 'final_model_PPO.zip')}")
        if args.algo == 'PPO':
            model = PPO.load(
                os.path.join(model_dir, "final_model_PPO"),
                env=env,
                tensorboard_log=log_dir,
                device=device
            )
        else:
            model = SAC.load(
                os.path.join(model_dir, "final_model_SAC"),
                env=env,
                tensorboard_log=log_dir,
                device=device
            )
    else:
        # Create a new model
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
                device=device,
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
                device=device,
            )
    
    # Train the model
    print(f"Training {args.algo} with multiple shapes for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name=f"{args.algo}_multiple_shapes",
    )
    
    # Save final model
    final_model_path = os.path.join(model_dir, f"ppo_multiple_shapes_500k")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Run 5 episodes with rendering for the final evaluation
    print("\nRunning 5 episodes with rendering for final evaluation...")
    render_env = SimpleRobotEnv(render_mode="human")
    
    # Apply the same observation adapter if needed
    if args.continue_training:
        render_env = ObservationAdapter(render_env, adaptation_mode='new_to_old')
    
    eval_rewards = []
    shape_counts = {"Box": 0, "Sphere": 0, "Cylinder": 0, "Unknown": 0}
    
    for episode in range(5):
        obs, _ = render_env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # Get the object shape for this episode
        current_shape = get_object_shape_name(render_env)
        shape_counts[current_shape] += 1
        
        print(f"Episode {episode+1}/5 with shape: {current_shape}")
        
        while not done:
            # Get action from the trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute the action
            obs, reward, terminated, truncated, _ = render_env.step(action)
            episode_reward += reward
            step += 1
            
            # Add small delay for visualization
            time.sleep(0.01)
            
            # Check if done
            done = terminated or truncated
            
            # Limit steps to avoid too long episodes
            if step >= 500:
                print(f"Episode reached maximum steps (500)")
                break
        
        eval_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {step}, Shape = {current_shape}")
    
    # Print evaluation summary
    print("\nEvaluation Summary:")
    print(f"Average reward: {np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}")
    print(f"Min reward: {np.min(eval_rewards):.2f}")
    print(f"Max reward: {np.max(eval_rewards):.2f}")
    
    # Print shape distribution
    print("\nShape Distribution:")
    for shape, count in shape_counts.items():
        if count > 0:
            print(f"{shape}: {count} episodes")
    
    # Close environments
    render_env.close()
    env.close()
    if env != eval_env:
        eval_env.close()

if __name__ == "__main__":
    main()