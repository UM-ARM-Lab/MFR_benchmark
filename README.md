# multi-finger reorientation benchmark 
The code adapts from https://github.com/UM-ARM-Lab/isaacgym-arm-envs
Author: Fan Yang
Version: 0.0.1

Install via `pip install -e` .

Dependencies:
- PyTorch
- Isaac Gym (https://developer.nvidia.com/isaac-gym)

## overview
This repository provides a Multi-Finger Dexterous Manipulation Benchmark for robotic hand environments, utilizing NVIDIA IsaacGym for physics simulation. The benchmark is designed to evaluate dexterous manipulation tasks such as valve turning, cuboid insertion, screwdriver turning, and object reorientation, among others. These tasks are implemented with the Allegro robotic hand and optional arm configurations.

## Features
- **Multiple Sensitive Manipulation Tasks**:
  - Valve turning (cylinder, cuboid, and cross-valves)
  - Screwdriver turning and manipulation
  - Cuboid insertion and alignment
  - Reorientation of complex objects
- **Customizable Environments**:
  - Flexible control modes (e.g., joint impedance, Cartesian impedance)
  - Gravity and friction adjustments
- **Simulation Tools**:
  - Built on NVIDIA IsaacGym for high-performance GPU-based simulation
  - Utilizes libraries such as PyTorch, PyTorch3D, and SciPy for advanced kinematics and transformations
- **Metrics and Validity Checks**:
  - Each environment includes state retrieval and task-specific validity checks.
