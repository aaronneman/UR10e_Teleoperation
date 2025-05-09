import openvr
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

# Configure logging
logging.basicConfig(level=logging.INFO)

class VRTracker:
    """Handles initialization and data retrieval from Vive trackers using OpenXR."""
    def __init__(self, simulation_mode=False):
        self.vr_system = None
        self.tracker_indices = []
        self.simulation_mode = simulation_mode
        self.initialized = False
        
        try:
            openvr.init(openvr.VRApplication_Other)
            self.vr_system = openvr.VRSystem()
            self.tracker_indices = self._get_tracker_indices()
            self.initialized = True
            if not self.tracker_indices:
                logging.warning("No Vive trackers detected.")
        except Exception as e:
            logging.error(f"Failed to initialize OpenVR: {e}")
            if not simulation_mode:
                raise
            else:
                logging.warning("Continuing in simulation mode without VR tracking.")

    def _get_tracker_indices(self):
        indices = []
        if self.vr_system is None:
            return indices
            
        try:
            for i in range(openvr.k_unMaxTrackedDeviceCount):
                if self.vr_system.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_GenericTracker:
                    indices.append(i)
        except Exception as e:
            logging.error(f"Error getting tracker indices: {e}")
        
        return indices

    def get_tracker_poses(self):
        """Returns a list of tuples (position, orientation) for each tracker."""
        poses = []
        
        # If not initialized or no trackers, return empty list
        if not self.initialized or not self.tracker_indices:
            return poses
            
        try:
            pose_data = self.vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
            )
            
            for idx in self.tracker_indices:
                pose = pose_data[idx]
                if not pose.bPoseIsValid:
                    poses.append(None)
                    continue
                matrix = pose.mDeviceToAbsoluteTracking
                # Extract position
                x = matrix[0][3]
                y = matrix[1][3]
                z = matrix[2][3]
                # Extract quaternion using scipy
                matrix_array = np.array([
                    [matrix[0][0], matrix[0][1], matrix[0][2]],
                    [matrix[1][0], matrix[1][1], matrix[1][2]],
                    [matrix[2][0], matrix[2][1], matrix[2][2]]
                ])
                rot = R.from_matrix(matrix_array)
                qx, qy, qz, qw = rot.as_quat()  # scipy returns x,y,z,w format
                poses.append(((x, y, z), (qw, qx, qy, qz)))
        except Exception as e:
            logging.error(f"Error getting   tracker poses: {e}")
            
        return poses

    def shutdown(self):
        if self.initialized:
            try:
                openvr.shutdown()
            except Exception as e:
                logging.error(f"Error shutting down OpenVR: {e}")
