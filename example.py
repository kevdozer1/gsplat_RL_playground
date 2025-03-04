#!/usr/bin/env python3
"""
A simple example script for the PLAYGROUND project.
This script creates a simple environment with a robot pushing a cube to a target.
"""

import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym

from playground_rl.environments.simple_robot_env import SimpleRobotEnv

def main():
    """Run a simple demonstration of the environment with random actions."""
    # Create the environment with rendering enabled
    env = SimpleRobotEnv(render_mode="human")
    
    # Reset the environment
    observation, info = env.reset()
    
    # Run for a number of steps with random actions
    for i in range(500):
        # Generate a random action
        action = env.action_space.sample()
        
        # Execute the action
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Print information
        print(f"Step {i}, Reward: {reward:.3f}")
        
        # Sleep to slow down the visualization
        time.sleep(0.01)
        
        # Reset if the episode is done
        if terminated or truncated:
            print("Episode finished. Resetting...")
            observation, info = env.reset()
    
    # Close the environment
    env.close()
    print("Simulation complete!")

if __name__ == "__main__":
    main()