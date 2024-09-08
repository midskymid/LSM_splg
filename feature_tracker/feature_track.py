import cv2
import rospy
import numpy as np
import torch
import time
from scipy import linalg
from utils.camera_models import PinholeCamera
from utils.tools import print_gpu_usage, numpy_image_to_torch

# For data alignment
def reduceVector(v, status):
    j = 0
    for i in range(v.shape[0]):
        if status[i]:
            v[j] = v[i]
            j += 1
    v = v[:j]
    return v

class FeatureTracker:
    def __init__(self, extractor, matcher, params) -> None:

        self.extractor = extractor
        self.matcher = matcher
        self.params = params
        
        self.mask = np.empty((0,0), dtype=np.float32)  # Use a grid with the same size as the latest image to remove feature points that are too close to each other. This helps in achieving a more uniform distribution of feature points and reduces unnecessary computation
        self.prev_img = np.empty((0,0,0), dtype=np.float32)  # The image from two frames ago
        self.cur_img = np.empty((0,0,0), dtype=np.float32)  # The previous frame image
        self.forw_img = np.empty((0,0,0), dtype=np.float32)  # The latest image
        self.resize_img = np.empty((0,0,0), dtype=np.float32)  # The latest image after resizing
        self.n_pts = np.empty((0,0), dtype=np.float32)  # size -> (N, 2)，Store feature point location information to be used in the addPoints function
        self.prev_pts = np.empty((0,0), dtype=np.float32)  # size -> (N, 2)，Store the feature point location information of the image from two frames ago
        self.cur_pts = np.empty((0,0), dtype=np.float32)  # size -> (N, 2)，Store the feature point location information of the previous frame
        self.forw_pts = np.empty((0,0), dtype=np.float32)  # size -> (N, 2)，Store the feature point location information of the latest frame
        self.prev_un_pts = np.empty((0,0), dtype=np.float32)  # size -> (N, 2)，Store the feature point location information on the normalized plane of the image from two frames ago
        self.cur_un_pts = np.empty((0,0), dtype=np.float32)  # size -> (N, 2)，Store the feature point location information on the normalized plane of the previous frame
        self.pts_velocity = np.empty((0,0), dtype=np.float32)  # size -> (N, 2)，Store the movement velocity of matched feature points in the pixel coordinate system between the previous frame and the frame from two frames ago
        self.ids = np.empty(0, dtype=np.int64)  # Store the ID of each feature point
        self.track_cnt = np.empty(0, dtype=np.int64)  # Store the number of matches for each feature point
        self.cur_un_pts_map = []  # Store the feature point IDs and their locations on the normalized plane for the previous frame
        self.prev_un_pts_map = []  # Store the feature point IDs and their locations on the normalized plane for the frame from two frames ago
        self.m_camera = PinholeCamera(params)   
        self.cur_time: float = None  # the timestamp of the previous frame image
        self.prev_time: float = None  # the timestamp of the frame from two frames ago

        self.n_id: int = 0  # This number is both the total number of all detected feature points and is used to assign a unique ID to each feature point
        self.n_max_cnt: int = 0  # The number of feature points that need to be supplemented
        self.resize_scale: tuple = None  # Image scaling ratio
        
        self.prev_desc = np.empty((0,0), dtype=np.float32)  # size -> (N, 64), Store the descriptors of the feature points from two frames ago
        self.cur_desc = np.empty((0,0), dtype=np.float32)  # size -> (N, 64), Store the descriptors of the feature points from the previous frame
        self.forw_desc = np.empty((0,0), dtype=np.float32)  # size -> (N, 64), Store the descriptors of the feature points from the latest frame
        self.n_desc = np.empty((0,0), dtype=np.float32)  # Store the descriptors of the feature points to be used in the addPoints function
        self.image_size = None  # size of image
        

    def inBorder(self, pt: np.ndarray) -> bool:
        """Determine whether it is a non-edge point; feature points located on the edge have poor tracking performance
           Inputs: 
           pt: np.ndarray: size -> (2,): feature point location on image
           Outputs:
           bool: if it is a non-edge point, return True; else, return False"""
        BORDER_SIZE = 1
        img_x = round(pt[0])
        img_y = round(pt[1])
        return BORDER_SIZE <= img_x and img_x < (self.params['image_width'] - BORDER_SIZE) and BORDER_SIZE <= img_y and img_y < (self.params['image_height'] - BORDER_SIZE)

    def readImage(self, _img: np.ndarray, _cur_time: float, params: dict):
        """Feature point extraction and matching, intra-class data transfer and processing
           Inputs: 
            _img: cv2.Mat: image data
            _cur_time: float: receive image data's time
            params: dict: parameters from config file"""
        # print_gpu_usage()
        self.params = params
        self.cur_time = _cur_time
        img = _img
        
        if self.forw_img is None:
            self.prev_img = self.cur_img = self.forw_img = img
        else:
            self.forw_img = img
        
        self.forw_pts = np.empty((0,0), dtype=np.float32)
        self.forw_desc = np.empty((0,0), dtype=np.float32)

        if np.size(self.cur_pts) > 2:
            # organize input of extractor and get outputs of extractor
            # torch_forw_img: torch.Tensor: size: (B, C, H, W) 

            torch_forw_img = numpy_image_to_torch(self.forw_img)
            
            forw_ptsdesc = self.extractor.extract(torch_forw_img)  # extract features
            self.image_size = forw_ptsdesc['image_size']

            tensor_cur_pts = torch.from_numpy(self.cur_pts).float()  # transfer np.ndarray to torch.Tensor
            tensor_cur_desc = torch.from_numpy(self.cur_desc).float()  
            _cur_pts = tensor_cur_pts.unsqueeze(0).to(self.params['device'])  # add batch dimension
            _cur_desc = tensor_cur_desc.unsqueeze(0).requires_grad_().to(self.params['device'])  # add batch dimension
            cur_ptsdesc = {
                'keypoints': _cur_pts,
                'descriptors': _cur_desc,
                'image_size': self.image_size
            }  # organize input of Lightglue
            
            src_matches = self.matcher({'image0': cur_ptsdesc, 'image1': forw_ptsdesc})  # descriptor matching
            
            feats0, feats1, matches01 = [rbd(x) for x in [cur_ptsdesc, forw_ptsdesc, src_matches]]  # remove batch dim
            kpts0, kdesc1, kpts1, matches = feats0['keypoints'].cpu().numpy(), feats1['descriptors'].detach().cpu().numpy(), feats1['keypoints'].cpu().numpy(), matches01['matches'].cpu().numpy()  # transfer torch.Tensor to np.ndarray
            
            # Initialize the matching status array, with 0 indicating unsuccessful tracking and 1 indicating successful tracking        
            status = np.zeros(kpts0.shape[0], dtype=np.int32)
            
            # Initialize the array for the latest frame's feature point location information
            self.forw_pts = np.zeros_like(kpts0)
            
            # Initialize the array for the latest frame descriptor information
            self.forw_desc = np.zeros_like(desc0)
            
            # Update the matching status array, the latest frame's feature point location information array, and the latest frame descriptor information array
            idx0 = matches[:,0]
            idx1 = matches[:,1]
            status[idx0] = 1
            self.forw_pts[idx0] = kpts1[idx1]
            self.forw_desc[idx0] = desc1[idx1]
            
            # The feature points at the edge positions of the image are set as unmatched for easy discarding by the subsequent reduceVector function
            for i in range(kpts0.shape[0]):
                if (status[i] and (not self.inBorder(self.forw_pts[i]))):
                    status[i] = 0
                
            # Discard the unmatched feature points information
            self.prev_pts = reduceVector(self.prev_pts, status)
            self.prev_pts = np.reshape(self.prev_pts, (-1, 2))
            self.prev_desc = reduceVector(self.prev_desc, status)

            self.cur_pts = reduceVector(self.cur_pts, status)
            self.cur_pts = np.reshape(self.cur_pts, (-1, 2))
            self.cur_desc = reduceVector(self.cur_desc, status)

            self.forw_pts = reduceVector(self.forw_pts, status)
            self.forw_pts = np.reshape(self.forw_pts, (-1, 2))
            self.forw_desc = reduceVector(self.forw_desc, status)

            self.ids = reduceVector(self.ids, status)
            self.cur_un_pts = reduceVector(self.cur_un_pts, status)
            self.track_cnt = reduceVector(self.track_cnt, status)

            
        self.track_cnt += 1  # Increment the tracking count for all features by 1

        if params['PUB_THIS_FRAME']:
            self.rejectWithF()  # "Reprojection check to verify if the matching is correct
            
            rospy.loginfo("set mask begins")
            self.setMask()  # Remove feature points that are too densely clustered in the image
            
            self.n_max_cnt = self.params['max_cnt'] - self.forw_pts.shape[0]
            # print("self.n_max_cnt size", self.n_max_cnt)
            if self.n_max_cnt > 0:
                if np.size(self.mask) == 0:
                    print("mask is empty ")
                if self.mask.dtype != np.uint8:
                    print("mask type wrong ")
                if self.mask.shape != self.forw_img.shape[:2]:
                    print("wrong size ")
                
                # extract kpts
                torch_forw_img = numpy_image_to_torch(self.forw_img)
                
                n_ptsdesc = self.extractor.extract(torch_forw_img.to(self.params['device']))
                self.image_size = n_ptsdesc['image_size']
                n_pts = n_ptsdesc['keypoints'].cpu().numpy()
                self.n_pts = np.reshape(n_pts, (-1, n_pts.shape[-1]))
                n_desc = n_ptsdesc['descriptors'].detach().cpu().numpy()
                self.n_desc = np.reshape(n_desc, (-1, n_desc.shape[-1]))
                                
                rospy.logdebug("add feature begins")
                
                self.addPoints()
                
        # information update        
        self.cur_pts = np.reshape(self.cur_pts, (-1, 2))
        self.forw_pts = np.reshape(self.forw_pts, (-1, 2))
        
        self.prev_img = self.cur_img
        self.prev_pts = self.cur_pts
        self.prev_desc = self.cur_desc
        self.prev_un_pts = self.cur_un_pts
        self.cur_img = self.forw_img
        self.cur_pts = self.forw_pts
        self.cur_desc = self.forw_desc
        self.undistortedPoints()
        self.prev_time = self.cur_time

        # --------test codes-----------
        # print("self.cur_un_pts.shape: ", self.cur_un_pts.shape)
        # print_gpu_usage()
        # --------test codes-----------

    # Remove feature points that are too close to each other and have fewer matches based on the number of times they have been matched                  
    def setMask(self):  
        self.mask = np.ones((self.params['image_height'], self.params['image_width']), dtype=np.uint8) * 255
        
        cnt_pts_id = []

        for i in range(self.forw_pts.shape[0]):
            cnt_pts_id.append((self.track_cnt[i], (self.forw_pts[i], self.ids[i], self.forw_desc[i])))
        cnt_pts_id.sort(reverse=True, key=lambda x: x[0])  # Sort based on the number of matches

        self.forw_pts = np.empty((0,0), dtype=np.float32)
        self.forw_desc = np.empty((0,0), dtype=np.float32)
        self.ids = np.empty(0, dtype=np.int64)
        self.track_cnt = np.empty(0, dtype=np.int64)
        self.n_pts = np.empty((0,0), dtype=np.float32)
        self.n_desc = np.empty((0,0), dtype=np.float32)

        for i, it in enumerate(cnt_pts_id):
            if self.mask[round(it[1][0][1]), round(it[1][0][0])] == 255:
                point = it[1][0]
                desc = it[1][2]
                if np.size(point) and np.size(desc):
                    point = np.reshape(point, (-1, point.size))
                    desc = np.reshape(desc, (-1, desc.size))

                    if np.size(self.forw_pts) == 0:
                        self.forw_pts = point
                    else:
                        self.forw_pts = np.vstack((self.forw_pts, point))

                    if np.size(self.forw_desc) == 0:
                        self.forw_desc = desc
                    else:
                        self.forw_desc = np.vstack((self.forw_desc, desc))

                    if np.size(self.ids) == 0:
                        self.ids = it[1][1]
                    else:
                        self.ids = np.append(self.ids, it[1][1])

                    if np.size(self.track_cnt) == 0:
                        self.track_cnt = it[0]
                    else:
                        self.track_cnt = np.append(self.track_cnt, it[0])

                    
                    cv2.circle(self.mask, [round(it[1][0][1]), round(it[1][0][0])], self.params['min_dist'], 0, -1)

            else:
                point = it[1][0]
                desc = it[1][2]
                point = np.reshape(point, (-1, point.size))
                desc = np.reshape(desc, (-1, desc.size))
             # Store the points that were successfully matched but removed due to being too close, so they can be added in the addPoints function to avoid extracting image feature point information again 
                if np.size(self.n_pts) == 0:
                    self.n_pts = point
                else:
                    self.n_pts = np.vstack((self.n_pts, point))
                
                if np.size(self.n_desc) == 0:
                    self.n_desc = desc
                else:
                    self.n_desc = np.vstack((self.n_desc, desc))

        # --------test codes-----------
        # print("----------------------")
        # print("self.forw_pts shape: ", self.forw_pts.shape)
        # print("self.track_cnt shape: ", self.track_cnt.shape)
        # print("self.forw_desc shape: ", self.forw_desc.shape)
        # print("self.ids shape: ", self.ids.shape)
        # print("self.n_pts shape: ", self.n_pts.shape)
        # --------test codes-----------

    # Insufficient number of feature points may affect subsequent steps like nonlinear optimization; add some new feature point information
    def addPoints(self):
        
        # more efficient than the old one
        if np.size(self.forw_pts) == 0 or np.size(self.forw_desc) == 0:
            self.forw_pts = self.n_pts[:self.n_max_cnt]
            self.forw_desc = self.n_desc[:self.n_max_cnt]

            n_ones = np.ones(self.n_max_cnt)
            self.ids = np.append(self.ids, -n_ones)
            self.track_cnt = np.append(self.track_cnt, n_ones)
        else:
            add_n_pts = self.n_pts[:self.n_max_cnt]
            add_n_desc = self.n_desc[:self.n_max_cnt]

            self.forw_pts = np.vstack((self.forw_pts, add_n_pts))
            self.forw_desc = np.vstack((self.forw_desc, add_n_desc))

            n_ones = np.ones(self.n_max_cnt)
            self.ids = np.append(self.ids, -n_ones)
            self.track_cnt = np.append(self.track_cnt, n_ones)

    # Number the newly added feature point information                 
    def updateID(self, i):
        if i < self.ids.shape[0]:
            if self.ids[i] == -1:
                self.ids[i] = self.n_id
                self.n_id += 1
            return True
        else:
            return False

    # Use the fundamental matrix to remove feature point information that does not satisfy the epipolar constraint
    def rejectWithF(self):
        if self.forw_pts.shape[0] >= 8:  # cv2.findFundamentalMat uses the eight-point algorithm.
            rospy.logdebug("FM ransac begins")
            # Project the feature point positions in the pixel coordinate system of the previous frame and the latest frame onto the normalized plane in the camera coordinate system
            un_cur_pts = np.zeros_like(self.cur_pts)
            un_forw_pts = np.zeros_like(self.forw_pts)
            for i in range(self.cur_pts.shape[0]):
                tmp_p = self.m_camera.liftProjective(self.cur_pts[i])
                un_cur_pts[i, 0] = tmp_p[0] / tmp_p[2]
                un_cur_pts[i, 1] = tmp_p[1] / tmp_p[2]

                tmp_p = self.m_camera.liftProjective(self.forw_pts[i])
                un_forw_pts[i, 0] = tmp_p[0] / tmp_p[2]
                un_forw_pts[i, 1] = tmp_p[1] / tmp_p[2]

            # Calculate the fundamental matrix and remove points with large reprojection errors
            F, status = cv2.findFundamentalMat(un_cur_pts, un_forw_pts, cv2.FM_RANSAC, self.params['F_threshold'], 0.99)
            size_a = self.cur_pts.shape[0]
            self.prev_pts = reduceVector(self.prev_pts, status)
            self.prev_desc = reduceVector(self.prev_desc, status)
            self.cur_pts = reduceVector(self.cur_pts, status)
            self.cur_desc = reduceVector(self.cur_desc, status)
            self.forw_pts = reduceVector(self.forw_pts, status)
            self.forw_desc = reduceVector(self.forw_desc, status)
            self.ids = reduceVector(self.ids, status)
            self.cur_un_pts = reduceVector(self.cur_un_pts, status)
            self.track_cnt = reduceVector(self.track_cnt, status)
            rospy.logdebug("FM ransac: %d -> %d: %f", size_a, self.forw_pts.shape[0], 1.0 * self.forw_pts.shape[0] / size_a)

    # Perform normalization plane projection and undistortion on the matched feature point location information
    def undistortedPoints(self):
        self.cur_un_pts = np.empty((0,0))
        self.cur_un_pts_map = []

        # --------test codes-----------
        # print("self.cur_pts size: ", np.size(self.cur_pts))
        # print("self.ids size: ", np.size(self.ids))
        # print("self.cur_pts shape: ", self.cur_pts.shape)
        # --------test codes-----------
        
        if np.size(self.ids) == 1:
            self.ids = np.reshape(self.ids, (-1))
        for i in range(self.cur_pts.shape[0]):
            a = self.cur_pts[i]
            b = self.m_camera.liftProjective(a)
            if np.size(self.cur_un_pts) == 0:
                self.cur_un_pts = np.array([b[0]/b[2], b[1]/b[2]])
            else:
                self.cur_un_pts = np.vstack((self.cur_un_pts, np.array([b[0]/b[2], b[1]/b[2]])))
                       
            self.cur_un_pts_map.append((self.ids[i], np.array([b[0]/b[2], b[1]/b[2]])))

        self.cur_un_pts = np.reshape(self.cur_un_pts, (-1, 2))
        
        if len(self.prev_un_pts_map) != 0:
            dt = self.cur_time - self.prev_time
            self.pts_velocity = np.empty((0,0))
            for i in range(self.cur_un_pts.shape[0]):
                if self.ids[i] != -1:
                    it_ids = self.ids[i]
                    it_index = 0
                    it = ()
                    for index, item in enumerate(self.prev_un_pts_map):
                        num, arr = item
                        if num == it_ids:
                            it = item
                            it_index = index + 1
                            break
                        it_index = index + 1
                    if it_index != len(self.prev_un_pts_map):

                        v_x = (self.cur_un_pts[i, 0] - it[1][0]) / dt
                        v_y = (self.cur_un_pts[i, 1] - it[1][1]) / dt
                        if np.size(self.pts_velocity) == 0:
                            self.pts_velocity = np.array([v_x, v_y])
                        else:
                            self.pts_velocity = np.vstack((self.pts_velocity, np.array([v_x, v_y])))
                    else:
                        if np.size(self.pts_velocity) == 0:
                            self.pts_velocity = np.array([0, 0])
                        else:
                            self.pts_velocity = np.vstack((self.pts_velocity, np.array([0, 0])))
                else:
                    if np.size(self.pts_velocity) == 0:
                        self.pts_velocity = np.array([0, 0])
                    else:
                        self.pts_velocity = np.vstack((self.pts_velocity, np.array([0, 0])))

        else:
            for i in range(self.cur_pts.shape[0]):
                if np.size(self.pts_velocity) == 0:
                    self.pts_velocity = np.array([0, 0])
                else:
                    self.pts_velocity = np.vstack((self.pts_velocity, np.array([0, 0])))

        
        self.pts_velocity = np.reshape(self.pts_velocity, (-1, 2))
        self.prev_un_pts_map = self.cur_un_pts_map
