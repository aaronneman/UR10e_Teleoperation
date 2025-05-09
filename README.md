# UR10 VR Teleoperation System

## Overview
This system provides VR-based teleoperation control for a Universal Robots UR10 robotic arm, with both simulation and real operation modes. It maps VR tracker movements to robot commands while handling coordinate system transformations and motion smoothing.

## Features
- VR tracker to robot motion mapping
- Support for both simulation and real operation modes
- Real-time visualization (optional)
- Automatic calibration procedure
- Configurable motion scaling and smoothing
- Continuous rotation tracking

## File Structure
- `Teleop_0.8.py`: Main teleoperation control program
- `RobotController.py`: Handles robot communication and command sending
- `MathUtils.py`: Mathematical utilities for quaternion operations
- `VRTracker.py`: VR tracking input interface
- `Visualization_0_1.py`: 3D visualization module
- `live_config.json`: Configuration file

## Installation
1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install numpy scipy transforms3d
   ```

## Configuration
Edit `live_config.json` with your settings:
```json
{
  "robot_ip": "your_robot_ip",
  "secondary_port": 30002,
  "home_pose": [-0.1450, 1.0750, 0.4840, 1.5525, 2.721, 2.5546],
  "scaling_factor": 1.0,
  "refresh_rate": 100.0,
  "smoothing_factor": 0.5,
  "simulation_mode": false,
  "visualization": false
}
```

## Usage
1. Run the program:
   ```bash
   python Teleop_0.8.py
   ```
2. Follow on-screen instructions for calibration
3. Move VR tracker to control robot

## Dependencies
- Python 3.7+
- numpy
- scipy
- transforms3d
- VR tracking system (specific implementation in VRTracker.py)

## File Descriptions

### Teleop_0.8.py
Main control program that:
- Handles VR tracker input
- Maps movements to robot commands
- Manages calibration
- Runs control loops for both simulation and real modes

### RobotController.py
Provides interface to UR10 robot with:
- TCP socket communication
- Command sending (movel, servol)
- Simulation mode support
- Automatic reconnection

### MathUtils.py
Mathematical utilities for:
- Quaternion operations
- Rotation conversions
- Vector transformations
- Interpolation

### Visualization_0_1.py
Optional 3D visualization showing:
- Robot position
- Tracker position
- Workspace limits

## License
[MIT License](LICENSE)
