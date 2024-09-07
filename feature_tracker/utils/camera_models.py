import numpy as np
import cv2
from .parameters import readParameters

class PinholeCamera(object):
    def __init__(self, params):
        self.proj_params = params['projection_parameters']
        self.distort_params = params['distortion_parameters']
        

    def liftProjective(self, p: np.ndarray) -> tuple:
        """Project points from the pixel coordinate system onto the normalized plane of the camera coordinate system and perform distortion correction.
           Inputs: size -> (2,) or (1, 2): Feature point positions in the pixel coordinate system.
           Outputs: (norm_x, norm_y, 1.0): Feature point positions on the normalized plane of the camera coordinate system.
        """
        fx = self.proj_params[0]
        fy = self.proj_params[1]
        cx = self.proj_params[2]
        cy = self.proj_params[3]
        k1 = self.distort_params[0]
        k2 = self.distort_params[1]
        p1 = self.distort_params[2]
        p2 = self.distort_params[3]
        
        camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
        distortion_coeffs = np.array([k1, k2, p1, p2])
        
        pixel_coord = p.reshape(-1, 2)
        
        undistorted_points = cv2.undistortPoints(pixel_coord, camera_matrix, distortion_coeffs)
        normalized_coord = undistorted_points[0][0]
        
        return (normalized_coord[0], normalized_coord[1], 1.0)
        

if __name__ == "__main__":
    pixel_coord = np.array([320, 240], dtype=np.float32)
    path = "C:\\Users\\16485\\Desktop\\LSM_ws\\src\\LSM\\config\\euroc\\AQUALOC_config.yaml"
    params = readParameters(path)
    pin_model = PinholeCamera(params)
    norm_pt = pin_model.liftProjective(pixel_coord)
    print(norm_pt)
    
    