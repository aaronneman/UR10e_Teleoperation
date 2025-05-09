import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
import logging
import numpy as np
from MathUtils import *

# Configure logging
logging.basicConfig(level=logging.INFO)

class RobotVisualizer:
    """Handles visualization of the robot end effector position and orientation 
    as well as the VR tracker position and orientation."""
    def __init__(self, robot_controller, workspace_limits=None):
        self.robot_controller = robot_controller
        self.fig = plt.figure(figsize=(12, 10))
        
        # Create a GridSpec layout with 2 rows, 2 columns
        # Top row spans both columns for position plot
        # Bottom row has tracker and robot orientation plots side by side
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
        
        # Create the 3D subplot for position and robot orientation (spans both columns)
        self.ax = self.fig.add_subplot(gs[0, :], projection='3d')
        
        # Create the 3D subplot for tracker orientation (bottom left)
        self.ax_tracker = self.fig.add_subplot(gs[1, 0], projection='3d')
        
        # Create the 3D subplot for robot end effector orientation (bottom right)
        self.ax_robot = self.fig.add_subplot(gs[1, 1], projection='3d')
        
        self.animation = None
        self.tracker_position = None  # Store the current tracker position
        self.initial_tracker_position = None  # Store the calibration position
        self.tracker_orientation = None  # Store the current tracker orientation
        
        # Set default workspace limits if not provided
        if workspace_limits is None:
            self.workspace_limits = {
                'x': (-0.5, 0.5),
                'y': (0.5, 1.5),
                'z': (0.0, 1.0)
            }
        else:
            self.workspace_limits = workspace_limits
        
        # Initialize plotting elements to None
        self.position_point = None
        self.tracker_point = None
        self.initial_tracker_point = None
        self.orientation_lines = []
        self.tracker_orientation_lines = []
        self.robot_orientation_lines = []
        
        # Setup the plot
        self._setup_plot()
        
    def _setup_plot(self):
        """Configure the 3D plots for robot and tracker visualization."""
        # Setup main 3D plot for robot and position
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Robot and Tracker Position Visualization')
        
        # Set axis limits based on workspace
        self.ax.set_xlim(self.workspace_limits['x'])
        self.ax.set_ylim(self.workspace_limits['y'])
        self.ax.set_zlim(self.workspace_limits['z'])
        
        # Add grid
        self.ax.grid(True)
        
        # Setup tracker orientation plot
        self.ax_tracker.set_xlabel('X')
        self.ax_tracker.set_ylabel('Y')
        self.ax_tracker.set_zlabel('Z')
        self.ax_tracker.set_title('Tracker Orientation Visualization')
        
        # Set fixed axis limits for orientation plot
        self.ax_tracker.set_xlim(-1, 1)
        self.ax_tracker.set_ylim(-1, 1)
        self.ax_tracker.set_zlim(-1, 1)
        
        # Add grid
        self.ax_tracker.grid(True)
        
        # Setup robot end effector orientation plot
        self.ax_robot.set_xlabel('X')
        self.ax_robot.set_ylabel('Y')
        self.ax_robot.set_zlabel('Z')
        self.ax_robot.set_title('Robot End Effector Orientation Visualization')
        
        # Set fixed axis limits for orientation plot
        self.ax_robot.set_xlim(-1, 1)
        self.ax_robot.set_ylim(-1, 1)
        self.ax_robot.set_zlim(-1, 1)
        
        # Add grid
        self.ax_robot.grid(True)
        
        # Adjust spacing between subplots
        plt.tight_layout()

    def set_tracker_position(self, position):
        """Update the current tracker position."""
        self.tracker_position = position
        
    def set_initial_tracker_position(self, position):
        """Set the initial (calibration) tracker position."""
        self.initial_tracker_position = position
        
    def set_tracker_orientation(self, orientation):
        """Update the current tracker orientation."""
        self.tracker_orientation = orientation

    def update_visualization(self, frame):
        """Update the visualization with the latest robot pose and tracker position and orientation."""
        pose = self.robot_controller.get_last_pose()
        
        # Create a list to collect all artists for returning
        artists = []
        
        # Clear previous elements in main plot
        if self.position_point:
            self.position_point.remove()
            self.position_point = None
            
        if self.tracker_point:
            self.tracker_point.remove()
            self.tracker_point = None
            
        if self.initial_tracker_point:
            self.initial_tracker_point.remove()
            self.initial_tracker_point = None
            
        for line in self.orientation_lines:
            if line:
                line.remove()
        self.orientation_lines = []
        
        # Clear previous elements in tracker orientation plot
        for line in self.tracker_orientation_lines:
            if line:
                line.remove()
        self.tracker_orientation_lines = []
        
        # Clear previous elements in robot orientation plot
        for line in self.robot_orientation_lines:
            if line:
                line.remove()
        self.robot_orientation_lines = []
        
        # If we have a robot pose, draw it
        if pose is not None:
            x, y, z, rx, ry, rz = pose
            
            # Plot the end effector position (red)
            self.position_point = self.ax.scatter([x], [y], [z], color='r', s=100, label='Robot End Effector')
            artists.append(self.position_point)
            
            # Convert rotation vector to quaternion
            q = euler_to_quaternion(rx, ry, rz)
            
            # Calculate orientation axes
            axis_length = 0.1  # Length of the orientation arrows
            
            # X axis (red)
            x_axis = quaternion_rotate_vector(q, (axis_length, 0, 0))
            line = self.ax.plot([x, x + x_axis[0]], 
                              [y, y + x_axis[1]], 
                              [z, z + x_axis[2]], 
                              'r-', linewidth=2, label='X-Axis' if not self.orientation_lines else "")[0]
            self.orientation_lines.append(line)
            artists.append(line)
            
            # Y axis (green)
            y_axis = quaternion_rotate_vector(q, (0, axis_length, 0))
            line = self.ax.plot([x, x + y_axis[0]], 
                              [y, y + y_axis[1]], 
                              [z, z + y_axis[2]], 
                              'g-', linewidth=2, label='Y-Axis' if len(self.orientation_lines) == 1 else "")[0]
            self.orientation_lines.append(line)
            artists.append(line)
            
            # Z axis (blue)
            z_axis = quaternion_rotate_vector(q, (0, 0, axis_length))
            line = self.ax.plot([x, x + z_axis[0]], 
                              [y, y + z_axis[1]], 
                              [z, z + z_axis[2]], 
                              'b-', linewidth=2, label='Z-Axis' if len(self.orientation_lines) == 2 else "")[0]
            self.orientation_lines.append(line)
            artists.append(line)
        
        # Plot the tracker position if available (blue)
        if self.tracker_position:
            tx, ty, tz = self.tracker_position
            self.tracker_point = self.ax.scatter([tx], [ty], [tz], color='b', s=80, label='VR Tracker')
            artists.append(self.tracker_point)
        
        # Plot the initial tracker position if available (green)
        if self.initial_tracker_position:
            ix, iy, iz = self.initial_tracker_position
            self.initial_tracker_point = self.ax.scatter([ix], [iy], [iz], color='g', s=60, label='Calibration Position')
            artists.append(self.initial_tracker_point)
        
        # Add legend to main plot if we have any artists
        if artists and not self.ax.get_legend():
            handles, labels = self.ax.get_legend_handles_labels()
            # Only include labels that are not empty
            valid_handles = [h for h, l in zip(handles, labels) if l]
            valid_labels = [l for l in labels if l]
            if valid_handles:
                self.ax.legend(valid_handles, valid_labels, loc='upper right')
        
        # Draw tracker orientation in the second plot if available
        if self.tracker_orientation:
            # Origin for the orientation visualization (center of the plot)
            origin = [0, 0, 0]
            
            # Get the quaternion
            q = self.tracker_orientation
            
            # Calculate orientation axes
            axis_length = 0.5  # Length of the orientation arrows
            
            # X axis (red)
            x_axis = quaternion_rotate_vector(q, (axis_length, 0, 0))
            line = self.ax_tracker.plot([origin[0], origin[0] + x_axis[0]], 
                                      [origin[1], origin[1] + x_axis[1]], 
                                      [origin[2], origin[2] + x_axis[2]], 
                                      'r-', linewidth=3, label='X-Axis' if not self.tracker_orientation_lines else "")[0]
            self.tracker_orientation_lines.append(line)
            artists.append(line)
            
            # Y axis (green)
            y_axis = quaternion_rotate_vector(q, (0, axis_length, 0))
            line = self.ax_tracker.plot([origin[0], origin[0] + y_axis[0]], 
                                      [origin[1], origin[1] + y_axis[1]], 
                                      [origin[2], origin[2] + y_axis[2]], 
                                      'g-', linewidth=3, label='Y-Axis' if len(self.tracker_orientation_lines) == 1 else "")[0]
            self.tracker_orientation_lines.append(line)
            artists.append(line)
            
            # Z axis (blue)
            z_axis = quaternion_rotate_vector(q, (0, 0, axis_length))
            line = self.ax_tracker.plot([origin[0], origin[0] + z_axis[0]], 
                                      [origin[1], origin[1] + z_axis[1]], 
                                      [origin[2], origin[2] + z_axis[2]], 
                                      'b-', linewidth=3, label='Z-Axis' if len(self.tracker_orientation_lines) == 2 else "")[0]
            self.tracker_orientation_lines.append(line)
            artists.append(line)
            
            # Add legend to tracker orientation plot
            if self.tracker_orientation_lines and not self.ax_tracker.get_legend():
                handles, labels = self.ax_tracker.get_legend_handles_labels()
                # Only include labels that are not empty
                valid_handles = [h for h, l in zip(handles, labels) if l]
                valid_labels = [l for l in labels if l]
                if valid_handles:
                    self.ax_tracker.legend(valid_handles, valid_labels, loc='upper right')
        
        # Draw robot end effector orientation in the third plot if available
        if pose is not None:
            # Origin for the orientation visualization (center of the plot)
            origin = [0, 0, 0]
            
            # Get the robot orientation from the pose
            rx, ry, rz = pose[3], pose[4], pose[5]
            q_robot = euler_to_quaternion(rx, ry, rz)
            
            # Calculate orientation axes
            axis_length = 0.5  # Length of the orientation arrows
            
            # X axis (red)
            x_axis = quaternion_rotate_vector(q_robot, (axis_length, 0, 0))
            line = self.ax_robot.plot([origin[0], origin[0] + x_axis[0]], 
                                     [origin[1], origin[1] + x_axis[1]], 
                                     [origin[2], origin[2] + x_axis[2]], 
                                     'r-', linewidth=3, label='X-Axis' if not self.robot_orientation_lines else "")[0]
            self.robot_orientation_lines.append(line)
            artists.append(line)
            
            # Y axis (green)
            y_axis = quaternion_rotate_vector(q_robot, (0, axis_length, 0))
            line = self.ax_robot.plot([origin[0], origin[0] + y_axis[0]], 
                                     [origin[1], origin[1] + y_axis[1]], 
                                     [origin[2], origin[2] + y_axis[2]], 
                                     'g-', linewidth=3, label='Y-Axis' if len(self.robot_orientation_lines) == 1 else "")[0]
            self.robot_orientation_lines.append(line)
            artists.append(line)
            
            # Z axis (blue)
            z_axis = quaternion_rotate_vector(q_robot, (0, 0, axis_length))
            line = self.ax_robot.plot([origin[0], origin[0] + z_axis[0]], 
                                     [origin[1], origin[1] + z_axis[1]], 
                                     [origin[2], origin[2] + z_axis[2]], 
                                     'b-', linewidth=3, label='Z-Axis' if len(self.robot_orientation_lines) == 2 else "")[0]
            self.robot_orientation_lines.append(line)
            artists.append(line)
            
            # Add legend to robot orientation plot
            if self.robot_orientation_lines and not self.ax_robot.get_legend():
                handles, labels = self.ax_robot.get_legend_handles_labels()
                # Only include labels that are not empty
                valid_handles = [h for h, l in zip(handles, labels) if l]
                valid_labels = [l for l in labels if l]
                if valid_handles:
                    self.ax_robot.legend(valid_handles, valid_labels, loc='upper right')
        
        # If we have no artists, return an empty list to avoid animation errors
        if not artists:
            return []
            
        return artists
    
    def start_animation(self, interval=100):
        """Start the animation loop but don't show the plot yet."""
        # Initialize with a dummy frame to prevent errors
        initial_artists = self.update_visualization(0)
        
        self.animation = FuncAnimation(
            self.fig, 
            self.update_visualization,
            interval=interval, 
            blit=True,
            cache_frame_data=False,  # Avoid memory issues
            save_count=100  # Limit cache size
        )
    
    def show(self):
        """Show the plot window."""
        plt.show()
