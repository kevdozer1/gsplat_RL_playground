# CLAUDE.md - Guide for Agentic Assistants

## Commands
- Setup environment: `setup.bat` (Windows) or `bash setup_script.sh` (Linux)
- Train model: `python standalone_train.py [--algo PPO|SAC] [--timesteps N] [--render]`
- Run trained model: `python run_model.py --model path/to/model`
- Quick environment test: `python example.py`

## Code Style Guidelines
- **Imports**: Standard libs first, then third-party (numpy, torch, gymnasium), local modules last
- **Documentation**: Use docstrings for modules, classes, and functions
- **Types**: Use explicit type conversion for reward values and boolean returns
- **Naming**: Snake_case for variables/functions, PascalCase for classes
- **Error Handling**: Convert numpy values to Python primitives before returning
- **Config**: Use YAML files in configs/ directory for environment parameters
- **Rendering**: Support both "human" and "rgb_array" render modes
- **Environment Structure**: Follow gymnasium.Env interface with reset(), step(), render()
- **State & Rewards**: Document observation space and reward calculation clearly

- Project Overview: 
  - Phase 1: Simple RL tasks (reaching, pushing) with a robotic arm and basic shapes (spheres, cubes) in PyBullet.
  - Phase 2: Introduce pyramids, rods, and basic object segmentation; prepare for future Gaussian Splat integration.
  - Future: Real-world execution (Phase 3) and advanced capabilities (Phase 4).

- Phase 2 Objectives:
  1. Add pyramids and rods with randomized positions and sizes.
  2. Implement basic object segmentation by tracking objects with unique IDs or colors.
  3. Create a placeholder system for loading external 3D data (e.g., .ply files).
  4. Update RL tasks to handle multiple objects (e.g., push all objects to a target zone).

- Instructions for Claude Code:
  - Run commands within the `wsl_env` virtual environment.