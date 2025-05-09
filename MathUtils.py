"""
MathUtils - A collection of mathematical utilities for 3D transformations and quaternion operations.

This module provides functions for working with quaternions, rotation matrices, Euler angles,
and axis-angle representations. It uses numpy, scipy.spatial.transform.Rotation, and transforms3d
for efficient numerical operations.

Key Features:
- Quaternion normalization, multiplication, inversion
- Conversion between quaternions, rotation matrices, Euler angles, and axis-angle representations
- Vector rotation using quaternions
- Quaternion interpolation (SLERP)
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
import transforms3d as t3d

def quaternion_normalize(q: tuple) -> tuple:
    """
    Normalize a quaternion to unit length.
    
    Args:
        q: Input quaternion as (w, x, y, z) tuple
        
    Returns:
        Normalized quaternion as (w, x, y, z) tuple
        
    Note:
        Returns identity quaternion (1,0,0,0) if input norm is near zero
    """
    # Convert to numpy array for easier operations
    q_array = np.array(q)
    norm = np.linalg.norm(q_array)
    
    if norm < 1e-10:  # Avoid division by very small numbers
        return (1.0, 0.0, 0.0, 0.0)  # Return identity quaternion
    
    q_normalized = q_array / norm
    return tuple(q_normalized)

def quaternion_inverse(q: tuple) -> tuple:
    """
    Returns the inverse of a quaternion.
    
    Args:
        q: Input quaternion as (w, x, y, z) tuple
        
    Returns:
        Inverse quaternion as (w, x, y, z) tuple
        
    Note:
        For unit quaternions, inverse is same as conjugate
    """
    # Convert to scipy Rotation object
    # scipy uses (x, y, z, w) format, but our code uses (w, x, y, z)
    w, x, y, z = q
    q_scipy = np.array([x, y, z, w])
    
    # Create rotation object and get inverse
    rot = R.from_quat(q_scipy)
    inv_rot = rot.inv()
    
    # Convert back to our format (w, x, y, z)
    x, y, z, w = inv_rot.as_quat()
    return (w, x, y, z)

def quaternion_multiply(q1: tuple, q2: tuple) -> tuple:
    """
    Multiplies two quaternions (Hamilton product).
    
    Args:
        q1: First quaternion as (w, x, y, z) tuple
        q2: Second quaternion as (w, x, y, z) tuple
        
    Returns:
        Product quaternion as (w, x, y, z) tuple
        
    Note:
        Result is normalized to prevent numerical drift
    """
    # Convert to transforms3d format (w, x, y, z)
    # transforms3d already uses (w, x, y, z) format which matches our code
    q1_array = np.array(q1)
    q2_array = np.array(q2)
    
    # Use transforms3d for quaternion multiplication
    result = t3d.quaternions.qmult(q1_array, q2_array)
    
    # Normalize to prevent drift
    return quaternion_normalize(tuple(result))

def quaternion_to_axis_angle_vec(q: tuple) -> tuple:
    """
    Converts a quaternion to a rotation vector (axis-angle representation).
    
    Args:
        q: Input quaternion as (w, x, y, z) tuple
        
    Returns:
        Rotation vector as (x, y, z) tuple where magnitude is rotation angle in radians
        
    Note:
        Returns zero vector for near-zero rotations
    """
    # Convert to scipy Rotation object
    w, x, y, z = q
    q_scipy = np.array([x, y, z, w])
    
    # Create rotation object and get rotation vector
    rot = R.from_quat(q_scipy)
    rot_vec = rot.as_rotvec()
    
    # Check for very small rotation angles
    angle = np.linalg.norm(rot_vec)
    if angle < 1e-10:
        # Return zero vector for near-zero rotations
        return (0.0, 0.0, 0.0)
    
    return tuple(rot_vec)

def rotation_matrix_to_quaternion(matrix):
    """Converts a 3x3 rotation matrix to a quaternion."""
    # Convert to numpy array
    matrix_array = np.array(matrix)
    
    # Use scipy to convert rotation matrix to quaternion
    rot = R.from_matrix(matrix_array)
    x, y, z, w = rot.as_quat()
    
    # Return in our format (w, x, y, z)
    return (w, x, y, z)

def quaternion_to_euler(q, original_euler=None, isDegrees=False):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        q: Quaternion in (w, x, y, z) format
        original_euler: Optional original Euler angles to match range
        isDegrees: If True, return angles in degrees, otherwise in radians
    
    Returns:
        Tuple of Euler angles (roll, pitch, yaw)
    """
    # Convert to scipy Rotation object
    w, x, y, z = q
    q_scipy = np.array([x, y, z, w])
    
    # Create rotation object and get Euler angles
    rot = R.from_quat(q_scipy)
    
    # Get Euler angles in 'xyz' convention (roll, pitch, yaw)
    euler = rot.as_euler('xyz', degrees=isDegrees)
    
    return tuple(euler)

def euler_to_quaternion(roll, pitch, yaw, isDegrees=False):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion.
    
    Args:
        roll: Roll angle
        pitch: Pitch angle
        yaw: Yaw angle
        isDegrees: If True, angles are in degrees, otherwise in radians
    
    Returns:
        Quaternion in (w, x, y, z) format
    """
    # Create scipy Rotation object from Euler angles
    rot = R.from_euler('xyz', [roll, pitch, yaw], isDegrees)
    
    # Get quaternion
    x, y, z, w = rot.as_quat()
    
    # Return in our format (w, x, y, z)
    return (w, x, y, z)

def apply_rotation_matrix_to_quaternion(quat_wxyz, rot_matrix):
    """
    Applies a rotation (given by a 3x3 matrix) to a quaternion.
    
    Args:
        quat_wxyz: Tuple/list of quaternion (w, x, y, z)
        rot_matrix: 3x3 numpy array, rotation matrix

    Returns:
        Tuple (w, x, y, z): The rotated quaternion, same convention as input.
    """
    w, x, y, z = quat_wxyz
    # Scipy uses (x, y, z, w)
    quat_scipy = np.array([x, y, z, w])

    # Create Rotation objects
    rot_align = R.from_matrix(rot_matrix)
    rot_quat = R.from_quat(quat_scipy)

    # Compose rotations: rot_align * rot_quat
    # In scipy's convention, this means rot_align is applied first, then rot_quat
    # (Rotation composition is read right-to-left)
    composed = rot_align * rot_quat

    # Convert back: (x, y, z, w)
    x2, y2, z2, w2 = composed.as_quat()
    return (w2, x2, y2, z2)  # back to (w, x, y, z)

def axis_angle_to_quaternion(rx, ry, rz):
    """Convert UR axis-angle [rx, ry, rz] to quaternion (w, x, y, z)."""
    rotvec = np.array([rx, ry, rz])
    rot = R.from_rotvec(rotvec)
    x, y, z, w = rot.as_quat()
    return (w, x, y, z)

def quaternion_rotate_vector(q, v):
    """Rotate a 3D vector v by quaternion q."""
    # Convert to transforms3d format
    q_array = np.array(q)
    v_array = np.array(v)
    
    # Use transforms3d to rotate vector
    rotated_v = t3d.quaternions.rotate_vector(v_array, q_array)
    
    return tuple(rotated_v)

def rotation_vector_to_rpy(rotation_vector):
  """
  Converts a rotation vector to roll, pitch, yaw angles.

  Args:
    rotation_vector (numpy.ndarray): A 3-element numpy array representing the rotation vector.

  Returns:
    tuple: A tuple containing roll, pitch, and yaw angles in radians, in that order.
  """
  rotation = R.from_rotvec(rotation_vector)
  rpy = rotation.as_euler('xyz', degrees=False)
  return rpy[0], rpy[1], rpy[2]

def interpolate_quaternions(q1, q2, t):
    """Interpolate between two quaternions using spherical linear interpolation (SLERP)."""
    # Convert to numpy arrays and normalize
    q1_array = np.array(q1)
    q2_array = np.array(q2)
    
    # Normalize the quaternions
    q1_norm = q1_array / np.linalg.norm(q1_array)
    q2_norm = q2_array / np.linalg.norm(q2_array)
    
    # Calculate the dot product
    dot = np.sum(q1_norm * q2_norm)
    
    # If the dot product is negative, negate one quaternion to take the shorter path
    if dot < 0.0:
        q2_norm = -q2_norm
        dot = -dot
    
    # Clamp dot to valid range
    dot = min(dot, 1.0)
    
    # Calculate the interpolation angle
    theta = np.arccos(dot)
    
    # If the angle is very small, use linear interpolation to avoid divide-by-zero
    if abs(theta) < 1e-6:
        result = q1_norm * (1.0 - t) + q2_norm * t
    else:
        # Perform SLERP
        sin_theta = np.sin(theta)
        s0 = np.sin((1.0 - t) * theta) / sin_theta
        s1 = np.sin(t * theta) / sin_theta
        result = q1_norm * s0 + q2_norm * s1
    
    # Normalize the result to prevent drift
    return tuple(result / np.linalg.norm(result))
