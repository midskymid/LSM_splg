import rospy
import torch
import cv2
import numpy as np
import tqdm
import time

from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import ChannelFloat32
from geometry_msgs.msg import Point32
from std_msgs.msg import Bool
from cv_bridge import CvBridge

from feature_track import FeatureTracker
from utils.parameters import readParameters
from utils.superpoint_open import SuperPointOpen
from utils.lightglue import LightGlue
from utils.tools import numpy_image_to_torch
from typing import Optional

pub_img: Optional[rospy.Publisher] = None  # publish /feature_tracker/feature topic，topic type (Point32)
pub_match: Optional[rospy.Publisher] = None  # Publish image data with keypoints，topic type (Image)
pub_restart: Optional[bool]  # indicating whether there is an error in publishing/subscribing image messages
trackerData: Optional[FeatureTracker] = None  # object of the class FeatureTracker
first_image_time: float = 0.0  # received first image time 
pub_count: int = 1  # count how many images have been published
first_image_flag: bool = True
last_image_time: float = 0.0
init_pub: bool = False

# image callback function
def img_callback(img_msg, params):
    global pub_img
    global pub_match
    global pub_restart
    global trackerData
    global first_image_time  # Each frame corresponds to a timestamp at intervals of delta_t = 1/FREQ
    global pub_count  # The number of frames continuously (without interruption/error) published within intervals of delta_t = 1/FREQ
    global first_image_flag  # False: This is not the first frame, True: This is the first frame
    global last_image_time  # The timestamp of the current frame or the previous frame
    global init_pub  # pub matching points or not

    params = params[0]  # Because in ROS, message handling functions require parameters to be passed in tuple form  

    if first_image_flag:
        first_image_flag = False
        first_image_time = img_msg.header.stamp.to_sec()  # get timestamp, type(float)
        last_image_time = img_msg.header.stamp.to_sec()  
        return
    
    # "Check if the obtained image message contains errors; if so, publish an error flag
    if img_msg.header.stamp.to_sec() - last_image_time > 1.0 or img_msg.header.stamp.to_sec() < last_image_time :
        rospy.logwarn("image discontinue! reset the feature tracker!")
        first_image_flag = True
        last_image_time = 0
        pub_count = 1
        restart_flag = Bool()
        restart_flag.data = True
        pub_restart.publish(restart_flag)
        return
    
    last_image_time = img_msg.header.stamp.to_sec()

    # publisher frequency control 
    if round(1.0 * pub_count / (img_msg.header.stamp.to_sec() - first_image_time)) <= params["freq"]:
        params['PUB_THIS_FRAME'] = True
        # reset the frequency control
        if abs(1.0 * pub_count / (img_msg.header.stamp.to_sec() - first_image_time) - params["freq"]) < 0.01 * params["freq"]:
            first_image_time = img_msg.header.stamp.to_sec()
            pub_count = 0

    else:
        params['PUB_THIS_FRAME'] = False

    # get and process image
    bridge = CvBridge()
    try:
        ptr = bridge.imgmsg_to_cv2(img_msg, "mono8")
    except Exception as e:
        print("convert ROS image message to OpenCV format(GRAY) image Error!")
        return
    
    show_img = ptr.copy()

    rospy.logdebug('processing camera')
    trackerData.readImage(ptr, img_msg.header.stamp.to_sec(), params)
    id_i = 0

    # update trackData's ids
    while True:
        completed = False
        completed |= trackerData.updateID(id_i) 
        id_i += 1
        if not completed:
            break
    
    if params['PUB_THIS_FRAME']:
        pub_count += 1  
        
        feature_points = PointCloud()
        id_of_point = ChannelFloat32()
        u_of_point = ChannelFloat32()
        v_of_point = ChannelFloat32()
        velocity_x_of_point = ChannelFloat32()
        velocity_y_of_point = ChannelFloat32()

        
        feature_points.header = img_msg.header
        feature_points.header.frame_id = "world"
        un_pts = trackerData.cur_un_pts
        cur_pts = trackerData.cur_pts
        ids = trackerData.ids
        pts_velocity = trackerData.pts_velocity

        # print(un_pts.shape[0]==ids.shape[0])
        
        for j in range(ids.shape[0]):
            p = Point32()
            p.x = un_pts[j,0]
            p.y = un_pts[j,1]
            p.z = 1

            feature_points.points.append(p)
            id_of_point.values.append(ids[j])
            u_of_point.values.append(cur_pts[j,0])
            v_of_point.values.append(cur_pts[j,1])
            velocity_x_of_point.values.append(pts_velocity[j,0])
            velocity_y_of_point.values.append(pts_velocity[j,1])

        feature_points.channels.append(id_of_point)
        feature_points.channels.append(u_of_point)
        feature_points.channels.append(v_of_point)
        feature_points.channels.append(velocity_x_of_point)
        feature_points.channels.append(velocity_y_of_point)

        rospy.logdebug("publish %f, at %f", feature_points.header.stamp.to_sec(), rospy.Time.now().to_sec())

        # If it's the first time processing the image message, don't publish. Feature point matching can only be achieved after two frames
        if not init_pub:
            init_pub = True
        else:
            pub_img.publish(feature_points)

        # If you choose to display the feature matching image
        if params['show_track']:
            tmp_img = show_img
            bridge2 = CvBridge()
            if trackerData.cur_pts.shape[0] != 0:
                if np.size(trackerData.track_cnt) == 1:
                    trackerData.track_cnt = np.reshape(trackerData.track_cnt, (-1))
                for j in range(trackerData.cur_pts.shape[0]):
                    len = min(1.0, 1.0 * trackerData.track_cnt[j] / 20)
                    cv2.circle(tmp_img, [round(trackerData.cur_pts[j, 0]), round(trackerData.cur_pts[j, 1])], 2, (round(255 * (1 - len)), 0, round(255 * len)), 2)
                tmp_img_msg = bridge2.cv2_to_imgmsg(tmp_img, 'mono8')

                pub_match.publish(tmp_img_msg)

    rospy.loginfo("whole feature tracker processing done")
    
def warm_up(extractor, matcher, test_image_path0: str, test_image_path1: str):
    """warm up
       Inputs: extractor: torch model; matcher: torch model; test_image_path0: str; test_image_path1: str"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    test_img0 = cv2.imread(test_image_path0, cv2.IMREAD_GRAYSCALE)
    test_img1 = cv2.imread(test_image_path1, cv2.IMREAD_GRAYSCALE)
    
    torch_forw_img0 = numpy_image_to_torch(test_img0)
    torch_forw_img1 = numpy_image_to_torch(test_img1)
    consume_times = []

    for i in tqdm.tqdm(range(100)):
        start_time = time.time()
        
        # image infer
        ptsdesc0 = extractor(torch_forw_img0[None].to(device))
        ptsdesc1 = extractor(torch_forw_img1[None].to(device))
            
       # matcher infer
        matches = matcher({"image0": ptsdesc0, "image1": ptsdesc1})
        
        consume_times.append(time.time() - start_time)

    
    
    print(f"Average FPS per image: {1/np.mean(consume_times):.4f}")
    print("finished warm up...")
    
    
def main(config_path):
    global pub_img  
    global pub_match  
    global pub_restart  
    global trackerData

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # select devices
    
    test_image_path0 = "assets/DSC_0410.JPG"
    test_image_path1 = "assets/DSC_0411.JPG"  

    rospy.init_node("feature_tracker", log_level=rospy.INFO)  # Set the node name and output message level
    
    # The function to read parameters and return a dictionary of parameters
    params = readParameters(config_path)
    params['device'] = device  

    # Load a model for extracting feature points from images
    extractor = SuperPointOpen(max_num_keypoints=params['max_cnt'], model_path=params['sp_uw_model_path']).eval().to(device)  
    
    # Load a model for feature point matching
    matcher = LightGlue(features='superpoint', model_path=params['sp_lg_model_path']).eval().to(device)
    
    # warm up
    warm_up(extractor, matcher, test_image_path0, test_image_path1)  
    
    # The class used to process subscribed image data
    trackerData  = FeatureTracker(extractor, matcher, params)  

    # Subscription function to subscribe to image data information
    sub_img = rospy.Subscriber(params['image_topic'], Image, img_callback, callback_args=(params,), queue_size=100)  
    
    # Publish a topic to send data obtained from images for backend processing
    pub_img = rospy.Publisher("/feature_tracker/feature", PointCloud, queue_size=1000) 
    
    # Publish a topic to send the image with drawn feature points
    pub_match = rospy.Publisher("/feature_tracker/feature_img", Image, queue_size=1000) 
    
    # If the obtained image information is incorrect, publish an error flag topic
    pub_restart = rospy.Publisher("/feature_tracker/restart", Bool, queue_size=1000)

    rospy.spin()

if __name__ == "__main__":
    config_path = "/home/midsky/LSM_sp_lg_ws/src/LSM_splg/config/euroc/AQUALOC_config.yaml"
    main(config_path)
    

