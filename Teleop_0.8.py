"""
Teleop_0.8 - Main teleoperation control program for UR10 robotic arm.

This module provides VR-based teleoperation control for a Universal Robots UR10 arm,
with both simulation and real operation modes. It maps VR tracker movements to robot
commands while handling coordinate system transformations and motion smoothing.

Key Features:
- VR tracker to robot motion mapping
- Support for both simulation and real operation modes
- Real-time visualization (optional)
- Automatic calibration procedure
- Configurable motion scaling and smoothing
- Continuous rotation tracking

Dependencies:
- MathUtils: For quaternion and rotation operations
- RobotController: For robot communication
- VRTracker: For VR tracking input
- Visualization_0_1: For 3D visualization (optional)
"""

import time
import logging
import json
import numpy as np
from typing import Tuple, Optional
from MathUtils import *
from scipy.spatial.transform import Rotation as R
from Visualization_0_1 import RobotVisualizer
from VRTracker import VRTracker
from RobotController import RobotController

# Configure logging
logging.basicConfig(level=logging.INFO)


def calibrate(vr_tracker: VRTracker, robot_controller: RobotController, 
             home_pose: Tuple[float, float, float, float, float, float]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Perform calibration between VR tracker and robot positions.
    
    Args:
        vr_tracker: VRTracker instance for getting tracker poses
        robot_controller: RobotController instance for moving robot
        home_pose: Tuple of (x,y,z,rx,ry,rz) defining robot's home position
        
    Returns:
        Tuple of (initial_position, initial_orientation) where:
        - initial_position: (x,y,z) tracker position at calibration
        - initial_orientation: (w,x,y,z) tracker quaternion at calibration
        
    Raises:
        RuntimeError: If no valid tracker pose is found during calibration
    """
    # Move robot to its home position
    logging.info("Moving robot to home position...")
    robot_controller.move_to_home(home_pose)
    
    # Always perform physical calibration with tracker
    input("Place the tracker at the robot's home position and press Enter to calibrate...")
    poses = vr_tracker.get_tracker_poses()
    
    # Check if we got valid tracker data
    if poses and poses[0] is not None:
        initial_position, initial_orientation = poses[0]
        logging.info("Physical calibration completed successfully.")
    else:
        # Always require valid tracker data
        raise RuntimeError("No valid tracker pose found during calibration.")
    
    logging.info("Calibration complete.")
    return initial_position, initial_orientation

def run_simulation_control_loop(
    vr_tracker: VRTracker,
    robot_controller: RobotController,
    visualizer: RobotVisualizer,
    initial_position: Tuple[float, float, float],
    initial_orientation: Tuple[float, float, float, float],
    home_pose: Tuple[float, float, float, float, float, float],
    scaling_factor: float,
    smoothing_factor: float,
    refresh_rate: float,
    previous_robot_pose: Tuple[float, float, float, float, float, float]
) -> Tuple[float, float, float, float, float, float]:
    """
    Run the main simulation control loop that maps VR tracker movements to robot commands.
    
    Args:
        vr_tracker: VRTracker instance for getting tracker poses
        robot_controller: RobotController instance for sending commands
        visualizer: RobotVisualizer instance for 3D visualization
        initial_position: (x,y,z) tracker position at calibration
        initial_orientation: (w,x,y,z) tracker quaternion at calibration
        home_pose: (x,y,z,rx,ry,rz) robot's home position and orientation
        scaling_factor: Scaling factor for position movements
        smoothing_factor: Smoothing factor (0-1) for motion filtering
        refresh_rate: Control loop frequency in Hz
        previous_robot_pose: Last commanded robot pose for smoothing
        
    Returns:
        Tuple of (x,y,z,rx,ry,rz) representing the last commanded robot pose
        
    Note:
        - Uses quaternion math for accurate rotation handling
        - Implements continuous rotation tracking to avoid angle wrapping
        - Applies motion smoothing to reduce jerkiness
        - Handles coordinate system transformations between VR and robot frames
    """

    logging.info("Starting simulation control loop.")
    period = 1.0 / refresh_rate  # Time period between iterations

    # VR->robot frame mapping
    alignment_rotmat = [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]
    prev_robot_quat = None
    
    # Initialize continuous rotation tracking
    continuous_rotation = np.zeros(3)  # Track continuous rotation around each axis
    prev_delta_rotation = np.zeros(3)  # Previous frame's delta rotation
    
    try:
        while True:
            loop_start = time.time()
            logging.info("="*120)
            logging.info("="*120)
            try:
                poses = vr_tracker.get_tracker_poses()
                if not poses or poses[0] is None:
                    logging.warning("Tracker pose invalid. Skipping update.")
                    time.sleep(max(0.01, period - (time.time() - loop_start)))
                    continue

                current_position, current_orientation = poses[0]
                visualizer.set_tracker_position(current_position)
                visualizer.set_tracker_orientation(current_orientation)

                initial_orientation_norm = quaternion_normalize(initial_orientation)
                current_orientation_norm = quaternion_normalize(current_orientation)
                logging.info(f"initial_orientation_norm: \t{quaternion_to_axis_angle_vec(initial_orientation_norm)}")
                logging.info(f"current_orientation_norm: \t{quaternion_to_axis_angle_vec(current_orientation_norm)}")

                # --- Position mapping ---
                current_pos_array = np.array(current_position)
                initial_pos_array = np.array(initial_position)
                delta_pos_vr = current_pos_array - initial_pos_array

                # Replace with your real mapping!
                position_transform = ([
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0],
                ])
                transformed_delta = position_transform @ delta_pos_vr
                transformed_delta *= scaling_factor

                robot_x = home_pose[0] + transformed_delta[0]
                robot_y = home_pose[1] + transformed_delta[1]
                robot_z = home_pose[2] + transformed_delta[2]

                # --- Delta orientation in VR frame ---
                initial_inv = quaternion_inverse(initial_orientation_norm)
                delta_orientation = quaternion_multiply(initial_inv, current_orientation_norm)
                logging.info(f"initial_inv: \t\t\t{quaternion_to_axis_angle_vec(initial_inv)}")
                logging.info(f"delta_orientation: \t\t{quaternion_to_axis_angle_vec(delta_orientation)}")

                # --- Detect dominant axis of rotation ---
                # Convert delta orientation to axis-angle to analyze rotation axis
                delta_axis_angle = quaternion_to_axis_angle_vec(delta_orientation)
                delta_abs = np.abs(delta_axis_angle)
                dominant_axis = np.argmax(delta_abs)
                logging.info(f"delta_axis_angle: \t\t{delta_axis_angle}")
                logging.info(f"dominant_axis: \t\t{dominant_axis} (0=X, 1=Y, 2=Z)")
                
                # Get home orientation
                home_quat = axis_angle_to_quaternion(home_pose[3], home_pose[4], home_pose[5])
                logging.info(f"home_quat (axis-angle input): \t{(home_pose[3], home_pose[4], home_pose[5])}")
                logging.info(f"home_quat (axis-angle): \t{quaternion_to_axis_angle_vec(home_quat)}")
                logging.info(f"home_quat (quaternion): \t{home_quat}")
                
                # Use different approaches for different dominant axes
                if dominant_axis == 2:  # Z-axis is dominant
                    # For Z-axis, apply delta first, then alignment (better for Z-axis)
                    logging.info("Using Z-axis optimized approach (delta first, then alignment)")
                    robot_quat_vr = quaternion_multiply(home_quat, delta_orientation)
                    robot_quat = apply_rotation_matrix_to_quaternion(robot_quat_vr, alignment_rotmat)
                else:
                    # For X and Y axes, apply alignment first, then delta (better for Y-axis)
                    logging.info("Using X/Y-axis optimized approach (alignment first, then delta)")
                    aligned_delta = apply_rotation_matrix_to_quaternion(delta_orientation, alignment_rotmat)
                    logging.info(f"aligned_delta: \t\t{quaternion_to_axis_angle_vec(aligned_delta)}")
                    robot_quat = quaternion_multiply(home_quat, aligned_delta)

                logging.info(f"robot_quat_pre_filter (axis-angle): \t{quaternion_to_axis_angle_vec(robot_quat)}")

                # --- Filter (quaternion space, for smoothness) ---
                if prev_robot_quat is None:
                    prev_robot_quat = robot_quat
                else:
                    filter_strength = smoothing_factor
                    robot_quat = interpolate_quaternions(prev_robot_quat, robot_quat, 1.0 - filter_strength)
                    prev_robot_quat = robot_quat

                # --- Convert to axis-angle vector for UR ---
                # Use scipy to compute a rotation vector (axis * angle)
                scipy_quat = [robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]]
                current_rotvec = R.from_quat(scipy_quat).as_rotvec()
                
                # Calculate delta rotation from previous frame
                if prev_robot_quat is not None:
                    # Get the delta rotation between frames
                    delta_rotation = current_rotvec - prev_delta_rotation
                    
                    # Check for large jumps that might indicate a flip
                    for i in range(3):
                        if abs(delta_rotation[i]) > np.pi:
                            # If we detect a large jump, adjust it to be continuous
                            if delta_rotation[i] > 0:
                                delta_rotation[i] -= 2 * np.pi
                            else:
                                delta_rotation[i] += 2 * np.pi
                    
                    # Update continuous rotation
                    continuous_rotation += delta_rotation
                
                # Store current rotation for next frame
                prev_delta_rotation = current_rotvec.copy()
                
                # Use continuous rotation for robot control
                robot_rx, robot_ry, robot_rz = continuous_rotation.tolist()

                logging.info(f"robot_quat_final (axis-angle): \t{(robot_rx, robot_ry, robot_rz)}")

                current_robot_pose = np.array([robot_x, robot_y, robot_z, home_pose[3], home_pose[4], home_pose[5]]) #-robot_rx, -robot_ry, -robot_rz])

                # --- Optional: smooth pose vector ---
                prev_pose_array = np.array(previous_robot_pose)
                smoothed_pose = prev_pose_array * smoothing_factor + current_robot_pose * (1 - smoothing_factor)
                smoothed_pose = smoothed_pose.tolist()

                logging.info(f"Current robot pose (final): \t{smoothed_pose[3:]}")

                robot_controller.send_move_command(*smoothed_pose)
                previous_robot_pose = smoothed_pose

            except Exception as e:
                logging.error(f"Error in control loop iteration: {e}")
            
            elapsed = time.time() - loop_start
            sleep_time = max(0.001, period - elapsed)
            time.sleep(sleep_time)
    except Exception as e:
        logging.error(f"Control loop error: {e}")
        import traceback
        traceback.print_exc()
    
    return previous_robot_pose

def run_real_mode_control_loop(
    vr_tracker: VRTracker,
    robot_controller: RobotController, 
    visualizer: Optional[RobotVisualizer],
    initial_position: Tuple[float, float, float],
    initial_orientation: Tuple[float, float, float, float],
    home_pose: Tuple[float, float, float, float, float, float],
    scaling_factor: float,
    smoothing_factor: float,
    refresh_rate: float,
    previous_robot_pose: Tuple[float, float, float, float, float, float],
    visualization: bool
) -> Tuple[float, float, float, float, float, float]:
    """
    Run the main real robot control loop that maps VR tracker movements to robot commands.
    
    Args:
        vr_tracker: VRTracker instance for getting tracker poses
        robot_controller: RobotController instance for sending commands
        visualizer: Optional RobotVisualizer instance for 3D visualization
        initial_position: (x,y,z) tracker position at calibration
        initial_orientation: (w,x,y,z) tracker quaternion at calibration
        home_pose: (x,y,z,rx,ry,rz) robot's home position and orientation
        scaling_factor: Scaling factor for position movements
        smoothing_factor: Smoothing factor (0-1) for motion filtering
        refresh_rate: Control loop frequency in Hz
        previous_robot_pose: Last commanded robot pose for smoothing
        visualization: Whether visualization is enabled
        
    Returns:
        Tuple of (x,y,z,rx,ry,rz) representing the last commanded robot pose
        
    Note:
        - Uses same core logic as simulation mode but optimized for real robot control
        - Handles coordinate system transformations between VR and robot frames
        - Implements continuous rotation tracking to avoid angle wrapping
        - Applies motion smoothing to reduce jerkiness
        - Runs at fixed refresh rate for consistent performance
    """
    
    logging.info("Starting real-time control loop. Press Ctrl+C to exit.")
    period = 1.0 / refresh_rate  # Time per iteration

    # VR<->robot orientation mapping
    alignment_rotmat = [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]
    prev_robot_quat = None
    
    # Initialize continuous rotation tracking
    continuous_rotation = np.zeros(3)  # Track continuous rotation around each axis
    prev_delta_rotation = np.zeros(3)  # Previous frame's delta rotation

    try:
        while True:
            loop_start = time.time()
            #logging.info("=" * 140)
            #logging.info("=" * 140)
            
            # --- Get tracker pose ---
            poses = vr_tracker.get_tracker_poses()
            if not poses or poses[0] is None:
                logging.warning("Tracker pose invalid. Skipping update.")
                time.sleep(max(0.01, period - (time.time() - loop_start)))
                continue

            current_position, current_orientation = poses[0]

            if visualization:
                visualizer.set_tracker_position(current_position)
                visualizer.set_tracker_orientation(current_orientation)

            # --- Normalize orientations ---
            initial_orientation_norm = quaternion_normalize(initial_orientation)
            current_orientation_norm = quaternion_normalize(current_orientation)
            #logging.info(f"initial_orientation_norm (axis-angle): \t{quaternion_to_axis_angle_vec(initial_orientation_norm)}")
            #logging.info(f"current_orientation_norm (axis-angle): \t{quaternion_to_axis_angle_vec(current_orientation_norm)}")

            # --- Position mapping ---
            current_pos_array = np.array(current_position)
            initial_pos_array = np.array(initial_position)
            delta_pos_vr = current_pos_array - initial_pos_array

            # TODO: Update to match your position mapping.
            position_transform = ([
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0],
                ])
            transformed_delta = position_transform @ delta_pos_vr
            transformed_delta *= scaling_factor

            robot_x = home_pose[0] + transformed_delta[0]
            robot_y = home_pose[1] + transformed_delta[1]
            robot_z = home_pose[2] + transformed_delta[2]

            # --- Compute delta orientation in VR frame ---
            initial_inv = quaternion_inverse(initial_orientation_norm)
            delta_orientation = quaternion_multiply(initial_inv, current_orientation_norm)
            #logging.info(f"initial_inv (axis-angle): \t\t{quaternion_to_axis_angle_vec(initial_inv)}")
            #logging.info(f"delta_orientation (axis-angle): \t{quaternion_to_axis_angle_vec(delta_orientation)}")

            # --- Detect dominant axis of rotation ---
            # Convert delta orientation to axis-angle to analyze rotation axis
            delta_axis_angle = quaternion_to_axis_angle_vec(delta_orientation)
            delta_abs = np.abs(delta_axis_angle)
            dominant_axis = np.argmax(delta_abs)
            
            # Get home orientation
            home_quat = axis_angle_to_quaternion(home_pose[3], home_pose[4], home_pose[5])
            
            # Use different approaches for different dominant axes
            if dominant_axis == 2:  # Z-axis is dominant
                # For Z-axis, apply delta first, then alignment (better for Z-axis)
                robot_quat_vr = quaternion_multiply(home_quat, delta_orientation)
                robot_quat = apply_rotation_matrix_to_quaternion(robot_quat_vr, alignment_rotmat)
            else:
                # For X and Y axes, apply alignment first, then delta (better for Y-axis)
                aligned_delta = apply_rotation_matrix_to_quaternion(delta_orientation, alignment_rotmat)
                robot_quat = quaternion_multiply(home_quat, aligned_delta)
            #logging.info(f"robot_quat_pre_filter (axis-angle): \t{quaternion_to_axis_angle_vec(robot_quat)}")

            # --- Filter jitter in quaternion space ---
            if prev_robot_quat is None:
                prev_robot_quat = robot_quat
            else:
                filter_strength = smoothing_factor
                robot_quat = interpolate_quaternions(prev_robot_quat, robot_quat, 1.0 - filter_strength)
                prev_robot_quat = robot_quat

            # --- Convert to axis-angle for robot ---
            # Use scipy to compute a rotation vector (axis * angle)
            scipy_quat = [robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]]
            current_rotvec = R.from_quat(scipy_quat).as_rotvec()
            
            # Calculate delta rotation from previous frame
            if prev_robot_quat is not None:
                # Get the delta rotation between frames
                delta_rotation = current_rotvec - prev_delta_rotation
                
                # Check for large jumps that might indicate a flip
                for i in range(3):
                    if abs(delta_rotation[i]) > np.pi:
                        # If we detect a large jump, adjust it to be continuous
                        if delta_rotation[i] > 0:
                            delta_rotation[i] -= 2 * np.pi
                        else:
                            delta_rotation[i] += 2 * np.pi
                
                # Update continuous rotation
                continuous_rotation += delta_rotation
            
            # Store current rotation for next frame
            prev_delta_rotation = current_rotvec.copy()
            
            # Use continuous rotation for robot control
            robot_rx, robot_ry, robot_rz = continuous_rotation.tolist()
            #logging.info(f"robot_quat_final (axis-angle): \t{(robot_rx, robot_ry, robot_rz)}")

            # --- Compose and smooth full pose ---
            current_robot_pose = np.array([robot_x, robot_y, robot_z, home_pose[3], home_pose[4], home_pose[5]]) #-robot_rx, -robot_ry, -robot_rz])
            prev_pose_array = np.array(previous_robot_pose)
            smoothed_pose = prev_pose_array * smoothing_factor + current_robot_pose * (1 - smoothing_factor)
            smoothed_pose = smoothed_pose.tolist()
            #logging.info(f"Current robot pose (final): \t{smoothed_pose[3:]}")

            # --- Send to robot & update state ---
            robot_controller.send_move_command(*smoothed_pose)
            previous_robot_pose = smoothed_pose

            # --- Rate regulation ---
            elapsed = time.time() - loop_start
            sleep_time = max(0.001, period - elapsed)
            time.sleep(sleep_time)

    except Exception as e:
        logging.error(f"Control loop error: {e}")
        import traceback
        traceback.print_exc()

    return previous_robot_pose

def main() -> None:
    """
    Main entry point for the teleoperation program.
    
    Handles:
    - Configuration loading
    - VR tracker initialization
    - Robot controller setup
    - Visualization setup (if enabled)
    - Calibration procedure
    - Starting appropriate control loop (simulation or real mode)
    
    Configuration Parameters (from live_config.json):
    - robot_ip: IP address of robot controller
    - secondary_port: Port for robot communication
    - home_pose: Default home position (x,y,z,rx,ry,rz)
    - scaling_factor: Scaling for VR to robot movement mapping
    - refresh_rate: Control loop frequency in Hz
    - smoothing_factor: Motion smoothing factor (0-1)
    - simulation_mode: Whether to run in simulation
    - visualization: Whether to show 3D visualization
    
    Note:
    - Handles graceful shutdown on Ctrl+C
    - Cleans up resources in finally block
    - Supports both simulation and real operation modes
    """
    try:
        with open('live_config.json') as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error("Config files not found.")
        return

    robot_ip = config.get('robot_ip', 'localhost')
    secondary_port = config.get('secondary_port', 30002)
    home_pose = config.get('home_pose', [-0.1450, 1.0750, 0.4840, 1.5525, 2.721, 2.5546])
    scaling_factor = config.get('scaling_factor', 1.0)
    refresh_rate = config.get('refresh_rate', 100.0)
    smoothing_factor = config.get('smoothing_factor', 0.5)
    simulation_mode = config.get('simulation_mode', False)
    visualization = config.get('visualization', False)

    print(f"--------------------------------\nConfiguration Loaded:\n Robot IP: {robot_ip}\n Port: {secondary_port}\n Home Pose: {home_pose}\n Scaling Factor: {scaling_factor}\n Refresh Rate: {refresh_rate} Hz\n Smoothing Factor: {smoothing_factor}\n Simulation Mode: {simulation_mode}\n Visualization Mode: {visualization}\n--------------------------------")

    vr_tracker = None
    robot_controller = None
    visualizer = None
    try:
        # Initialize the robot controller first (works in both simulation and real mode)
        robot_controller = RobotController(robot_ip, secondary_port, simulation_mode=simulation_mode)
        
        # Set up visualization if in simulation mode
        if simulation_mode:
            logging.info("Simulation mode enabled. Setting up visualization...")
            # Define workspace limits for the visualizer
            workspace_limits = {
                'x': (-0.5, 0.5),
                'y': (0.5, 1.5),
                'z': (0.0, 1.0)
            }
            visualizer = RobotVisualizer(robot_controller, workspace_limits)
            
            # Start the animation
            visualizer.start_animation(interval=int(1000/refresh_rate))  # Convert refresh rate to milliseconds
            
            # In simulation mode, initialize VR tracker with simulation_mode=True to prevent crashes
            vr_tracker = VRTracker(simulation_mode=True)
            has_trackers = vr_tracker.initialized and len(vr_tracker.tracker_indices) > 0
            
            if not has_trackers:
                logging.warning("No VR trackers available in simulation mode. Running in demo mode.")
        
        elif not simulation_mode and visualization:
            logging.info("Visualization Enabled with realtime control. Setting up visualization...")
            # Define workspace limits for the visualizer
            workspace_limits = {
                'x': (-0.5, 0.5),
                'y': (0.5, 1.5),
                'z': (0.0, 1.0)
            }
            visualizer = RobotVisualizer(robot_controller, workspace_limits)
            
            # Start the animation
            visualizer.start_animation(interval=int(1000/refresh_rate))  # Convert refresh rate to milliseconds
            
            # In real mode, we need the VR tracker
            try:
                vr_tracker = VRTracker(simulation_mode=False)
                has_trackers = vr_tracker.initialized and len(vr_tracker.tracker_indices) > 0
                if not has_trackers:
                    logging.error("No trackers available. Exiting.")
                    return
            except Exception as e:
                logging.error(f"Failed to initialize VR tracker in real mode: {e}")
                return

        else:
            # In real mode, we need the VR tracker
            try:
                vr_tracker = VRTracker(simulation_mode=False)
                has_trackers = vr_tracker.initialized and len(vr_tracker.tracker_indices) > 0
                if not has_trackers:
                    logging.error("No trackers available. Exiting.")
                    return
            except Exception as e:
                logging.error(f"Failed to initialize VR tracker in real mode: {e}")
                return
        
        #Calibrate Arm and Tracker Positions
        initial_position, initial_orientation = calibrate(vr_tracker, robot_controller, home_pose)
        
        # Convert home pose orientation to quaternion for consistent math
        home_rx, home_ry, home_rz = home_pose[3], home_pose[4], home_pose[5]
        home_orientation = euler_to_quaternion(home_rx, home_ry, home_rz)
        
        logging.info("Calibration complete.")
        
        # Initialize previous pose for smoothing
        previous_robot_pose = home_pose
        
        # If we have a visualizer, set the initial positions
        if visualizer:
            visualizer.set_initial_tracker_position(initial_position)
            visualizer.set_tracker_orientation(initial_orientation)
        
        logging.info("Starting control loop. Press Ctrl+C to exit.")
        period = 1.0 / refresh_rate  # Time period between iterations
        
        # If in simulation mode and we have a visualizer, show it
        if simulation_mode and visualizer:
            logging.info("Running simulation visualization...")
            
            # Start a separate thread for the control loop if we have trackers
            if has_trackers:
                import threading
                
                # Start the simulation control loop in a separate thread
                def thread_wrapper():
                    nonlocal previous_robot_pose
                    previous_robot_pose = run_simulation_control_loop(
                        vr_tracker,
                        robot_controller,
                        visualizer,
                        initial_position,
                        initial_orientation,
                        home_pose,
                        scaling_factor,
                        smoothing_factor,
                        refresh_rate,
                        previous_robot_pose
                    )
                
                control_thread = threading.Thread(target=thread_wrapper, daemon=True)
                control_thread.start()
                logging.info("Started control loop in background thread")
            else:
                logging.info("Running in demo mode without trackers")
            
            # Show the visualization window - this will block the main thread
            visualizer.show()
            
            # When the visualization window is closed, we'll exit
            return
        
        elif not simulation_mode and visualizer and visualization:
            logging.info("Running realtime and visualization...")
            
            # Start a separate thread for the control loop if we have trackers
            if has_trackers:
                import threading
                
                # Start the simulation control loop in a separate thread
                def thread_wrapper():
                    nonlocal previous_robot_pose
                    previous_robot_pose = run_real_mode_control_loop(
                        vr_tracker,
                        robot_controller,
                        visualizer,
                        initial_position,
                        initial_orientation,
                        home_pose,
                        scaling_factor,
                        smoothing_factor,
                        refresh_rate,
                        previous_robot_pose,
                        visualization
                    )
                
                control_thread = threading.Thread(target=thread_wrapper, daemon=True)
                control_thread.start()
                logging.info("Started control loop in background thread")
            else:
                logging.info("Running in demo mode without trackers")
            
            # Show the visualization window - this will block the main thread
            visualizer.show()
            
            # When the visualization window is closed, we'll exit
            return

        # If we get here, we're in real mode with trackers
        # Run the normal control loop
        run_real_mode_control_loop(
            vr_tracker,
            robot_controller,
            visualizer,
            initial_position,
            initial_orientation,
            home_pose,
            scaling_factor,
            smoothing_factor,
            refresh_rate,
            previous_robot_pose,
            visualization
        )

    except KeyboardInterrupt:
        logging.info("Exiting.")
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if vr_tracker:
            vr_tracker.shutdown()
        if robot_controller:
            robot_controller.shutdown()
        # The visualizer will close when the program exits

if __name__ == "__main__":
    main()
