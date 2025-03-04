# GSPLAT RL Playground

A reinforcement learning environment for robotic manipulation tasks with multiple object shapes. This project serves as a foundation for exploring reinforcement learning for robotic manipulation in controllable environments.

## Overview

GSPLAT RL Playground is a large-scale RL-based robotic training system that uses segmented Gaussian Splat objects for training robotic manipulation tasks. The project follows a phased approach:

1. **Phase 1**: Simple RL tasks with multiple object shapes (current phase)
2. **Phase 2**: Add object segmentation and reconstruction (future)
3. **Phase 3**: Bridge to real-world execution (future)
4. **Phase 4**: Advanced capabilities (future)

## Features

- PyBullet-based physics simulation environment
- Multiple object shapes (Box, Sphere, Cylinder) with different visual properties
- Compatible with OpenAI Gymnasium interface
- Stable-Baselines3 integration for easy training and evaluation
- Configurable reward functions for reaching and pushing tasks
- Support for RGB rendering for visualization

## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/kevdozer1/gsplat_RL_playground.git
cd gsplat_RL_playground

# On Windows
setup.bat

# On Linux/Mac
source wsl_env/bin/activate
```

### Usage

#### Quick Environment Test

```bash
python example.py
```

#### Training a Model

```bash
python train_multiple_shapes.py --timesteps 200000 --num-envs 4
```

Additional training options:
- `--algo PPO|SAC`: Choose between PPO or SAC algorithm (default: PPO)
- `--continue-training`: Continue training from a previous model
- `--render`: Enable rendering during training
- `--cpu`: Force CPU use even if CUDA is available

#### Running a Trained Model

```bash
python run_model.py --model results/models/final_model_PPO.zip
```

Additional options:
- `--episodes N`: Number of episodes to run (default: 10)
- `--no-render`: Disable rendering
- `--algo PPO|SAC`: Specify the algorithm used for training

## Project Structure

- `configs/`: Configuration files
- `playground_rl/`: Main package
  - `environments/`: RL environments
    - `simple_robot_env.py`: Main environment with multiple object shapes
  - `models/`: Model definitions
  - `utils/`: Utility functions
- `results/`: Training results and saved models
- `example.py`: Simple example script
- `train_multiple_shapes.py`: Training script for multiple shapes environment
- `run_model.py`: Run and evaluate trained models

## Current Features (Phase 1)

- Simple robotic arm environment using PyBullet
- Multiple object shapes (Box, Sphere, Cylinder) with different visual properties
- Basic object manipulation (reaching and pushing)
- Training with PPO or SAC algorithms from Stable Baselines3
- Mesh visualization and loading utilities
- Configuration through YAML files

## Future Roadmap

### Phase 2: Incorporating Object Segmentation and Reconstruction
- Integrate vision pipeline for object segmentation
- Add Gaussian Splat reconstruction
- Spawn new object models in the simulation

### Phase 3: Bridging to Real World Execution
- Real robot setup and testing
- Sim-to-real transfer techniques
- System calibration

### Phase 4: Advanced Capabilities
- Multi-robot scenarios
- Generative environment variation
- Automated curriculum learning

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

- PyBullet for the physics simulation
- OpenAI Gymnasium for the RL environment interface
- Stable-Baselines3 for RL algorithms implementation