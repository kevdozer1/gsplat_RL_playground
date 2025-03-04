#!/bin/bash
# This script organizes the created files into the correct directory structure

# Create directory structure
mkdir -p playground_rl/{environments,models,utils,configs,tests,data/meshes}

# Make sure __init__.py files exist
touch playground_rl/__init__.py
touch playground_rl/environments/__init__.py
touch playground_rl/models/__init__.py
touch playground_rl/utils/__init__.py
touch playground_rl/tests/__init__.py

# Move environment file
cat > playground_rl/environments/simple_robot_env.py << 'EOL'
# Content of environment.py goes here
EOL

# Move training script
cat > playground_rl/train.py << 'EOL'
# Content of training-script.py goes here
EOL

# Move utils
cat > playground_rl/utils/mesh_utils.py << 'EOL'
# Content of mesh-utils.py goes here
EOL

# Move config
mkdir -p configs
cat > configs/default.yaml << 'EOL'
# Content of config-file.yaml goes here
EOL

# Move main script
cat > playground_rl/main.py << 'EOL'
# Content of main-script.py goes here
EOL

# Create a minimal example script for quickly testing
cat > example.py << 'EOL'
#!/usr/bin/env python3
"""
A minimal example script to quickly test the PLAYGROUND environment.
"""

from playground_rl.environments.simple_robot_env import SimpleRobotEnv
import time

# Create and test the environment
env = SimpleRobotEnv(render_mode="human")
observation, info = env.reset()

# Run for a few steps with random actions
for _ in range(100):
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.01)  # Slow down for visualization
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
print("Environment test completed successfully!")
EOL

# Make example.py executable
chmod +x example.py

# Create requirements.txt
cat > requirements.txt << 'EOL'
pybullet>=3.2.5
gymnasium>=0.28.1
stable-baselines3>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
torch>=2.0.0
trimesh>=3.20.0
PyYAML>=6.0
EOL

echo "File structure organized successfully!"