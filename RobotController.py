"""
RobotController - Handles communication and control of UR10 robotic arm.

This module provides a class for sending commands to a Universal Robots UR10 robotic arm
either in real mode (via TCP socket) or simulation mode (storing commands only).

Key Features:
- Persistent TCP connection to robot controller
- Support for both real and simulation modes
- Movement commands (servol and movel)
- Automatic reconnection on failure
- Logging of all operations

Command Format:
- Uses URScript language commands
- movel: Linear movement to specified pose
- servol: Servo movement with blending
"""
import socket
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)

class RobotController:
    """
    Controller for UR10 robotic arm with real and simulation modes.
    
    Provides methods for sending movement commands and managing robot connection.
    
    Args:
        robot_ip: IP address of the robot controller
        port: Port number for robot communication
        simulation_mode: If True, runs in simulation without actual robot connection
    """
    def __init__(self, robot_ip: str, port: int, simulation_mode: bool = False):
        self.robot_ip = robot_ip
        self.port = port
        self.socket = None
        self.simulation_mode = simulation_mode
        self.last_command = None
        
        if not simulation_mode:
            self._connect()

    def _connect(self):
        """Establish a persistent connection to the robot."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2)
            self.socket.connect((self.robot_ip, self.port))
            logging.info(f"Connected to {self.robot_ip}:{self.port}")
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            raise

    def send_move_command(self, x: float, y: float, z: float, rx: float, ry: float, rz: float) -> None:
        """
        Send a servol command to move robot to specified pose.
        
        Args:
            x: Target X position in meters
            y: Target Y position in meters
            z: Target Z position in meters
            rx: Rotation about X axis in radians
            ry: Rotation about Y axis in radians
            rz: Rotation about Z axis in radians
            
        Note:
            Uses servol command for smooth servo motion with blending
            Stores last command for visualization in simulation mode
        """
        command = f"servol(p[{x:.6f}, {y:.6f}, {z:.6f}, {rx:.9f}, {ry:.9f}, {rz:.9f}], 0.15)\n"        
        self.last_command = (x, y, z, rx, ry, rz)
        
        if not self.simulation_mode:
            self._send_command(command)

    def move_to_home(self, home_pose: Tuple[float, float, float, float, float, float]) -> None:
        """
        Move robot to predefined home position using movel command.
        
        Args:
            home_pose: Tuple containing (x, y, z, rx, ry, rz) of home position
                x,y,z: Position coordinates in meters
                rx,ry,rz: Rotation angles in radians
                
        Note:
            Uses movel command for precise linear movement to home position
            Stores last command for visualization in simulation mode
        """
        x, y, z, rx, ry, rz = home_pose
        self.last_command = (x, y, z, rx, ry, rz)
        
        if not self.simulation_mode:
            command = f"movel(p[{x:.6f}, {y:.6f}, {z:.6f}, {rx:.6f}, {ry:.6f}, {rz:.6f}])\n"
            self._send_command(command)

    def _send_command(self, command: str) -> None:
        """
        Internal method to send commands to robot via TCP socket.
        
        Args:
            command: URScript command string to send
            
        Note:
            Automatically reconnects if sending fails
            Skips sending in simulation mode
        """
        if self.simulation_mode:
            return
            
        try:
            self.socket.sendall(command.encode('utf-8'))
        except Exception as e:
            logging.error(f"Command failed: {e}, reconnecting...")
            self._connect()
            self.socket.sendall(command.encode('utf-8'))

    def get_last_pose(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Get the last commanded robot pose.
        
        Returns:
            Tuple of (x, y, z, rx, ry, rz) representing last commanded pose, or None if no commands sent
            x,y,z: Position coordinates in meters
            rx,ry,rz: Rotation angles in radians
        """
        return self.last_command

    def shutdown(self) -> None:
        """
        Cleanly shutdown the robot connection.
        
        Note:
            Only closes connection in real mode (no action in simulation)
            Logs shutdown event
        """
        if self.socket and not self.simulation_mode:
            self.socket.close()
            logging.info("Robot connection closed")
