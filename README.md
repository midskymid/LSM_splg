# LSM_splg
A Learning-based Visual-Inertial Odometry for Robust Position Measurement in Challenging Underwater Environments

# 1. Introduction
This repository implements the ROS2 version of VINS-MONO, mainly including the following packages:
* **camera_model**
* **feature_tracker**
* **vins_estimator**
* **pose_graph**
* **config**
 
# 2. Prerequisites
* System
  * Ubuntu 20.04.6  
  * ROS1 noetic
* Libraries
  * OpenCV 4.2.0
  * [Ceres Solver](http://ceres-solver.org/installation.html) 1.14.0
  * Eigen 3.3.7
# 3. Build LSM_splg
Clone the repository and catkin_make:  
```
cd $(PATH_TO_YOUR_ROS2_WS)/src
git clone https://github.com/midskymid/LSM_splg.git
cd ..
catkin_make
```
**PS: Don't forget to modify file paths of AQUALOC_config.yaml and feature_tracker_node.py.**
# 4. LSM_splg on AQUALOC datasets
## 4.1. ROS1 bag
Download [AQUALOC Archaeological datasets](https://seafile.lirmm.fr/d/79b03788f29148ca84e5/?p=%2FArchaeological_site_sequences&mode=list).   
## 4.2. Visual-inertial odometry and loop closure
All configuration files are in the package, **_config_**.  
Open four terminals, launch the feature_tracker, vins_estimator, rviz2, and ros2 bag. Take the MH01 for example
```
roslaunch vins_estimator vins_rviz.launch                               # for rviz
python feature_tracker_node.py                                          # for feature tracking
roslaunch vins_estimator lsm.launch                              # for backend optimization and loop 
rosbag play $(PATH_TO_YOUR_DATASET)/archaeo_sequence_1.bag              # for ros1 bag
```
# 5. Acknowledgements
We use ros1 version of [VINS MONO](https://github.com/HKUST-Aerial-Robotics/VINS-Mono),  [ceres solver](http://ceres-solver.org/installation.html) for non-linear optimization, [DBoW2](https://github.com/dorian3d/DBoW2) for loop detection. Also, we referred to parts of the implementations from [SuperPoint](https://github.com/rpautrat/SuperPoint) and (https://github.com/magicleap/SuperPointPretrainedNetwork),
[LightGlue](https://github.com/cvg/LightGlue) and (https://github.com/fabio-sim/LightGlue-ONNX)

# 6. Licence
The source code is released under [GPLv3](https://www.gnu.org/licenses/) license.

# LSM_splg_trt
A Learning-based Subaquatic Monocular Visual-Inertial Odometry Designed for the Jetson Platform coming soon...