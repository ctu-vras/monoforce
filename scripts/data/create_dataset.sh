#!/bin/bash

# This script creates a dataset from a bag file
# It saves point clouds and trajectories from a bag file
# It also adds calibrations: camera info and transforms

TIME_STEP=1.0
VIS=True
SAVE=False

# list of bag files in a directory
#BAG_PATHS='/media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/radarize__2024-02-07-10-47-13_0.bag
#           /media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/radarize__2024-02-07-10-48-58_1.bag
#           /media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/radarize__2024-02-07-10-51-05_2.bag
#           /media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/radarize__2024-02-07-10-52-55_3.bag
#           /media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/radarize__2024-02-07-10-54-50_4.bag
#           /media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/radarize__2024-02-07-10-57-02_5.bag
#           /media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/radarize__2024-02-07-10-59-07_6.bag
#           /media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/radarize__2024-02-07-11-00-59_7.bag
#           /media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/radarize__2024-02-07-11-02-34_8.bag
#           /media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/radarize__2024-02-07-11-04-24_9.bag
#           /media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/radarize__2024-02-07-11-06-24_10.bag
#           /media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/postproc/map2odom.bag
#           /media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/postproc/points.bag'
BAG_PATHS='/media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/radarize__2024-02-07-10-47-13_0.bag
           /media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/postproc/map2odom.bag
           /media/ruslan/SSD/data/bags/ORU/2024_02_07_Husky_campus_forest_bushes/bags/postproc/points.bag'
CLOUD_TOPIC='/point_cloud_deskewed'
LIDAR_FRAME='os_sensor'
CAMERA_TOPICS='/ids_camera/image_raw/compressed'
CAMERA_INFO_TOPICS='/ids_camera/camera_info'

# source ROS workspace
source /home/$USER/workspaces/traversability_ws/devel/setup.bash

# save sensor data
./save_sensor_data --bag-paths ${BAG_PATHS} \
                   --cloud-topics ${CLOUD_TOPIC} \
                   --lidar-frame ${LIDAR_FRAME} \
                   --camera-topics ${CAMERA_TOPICS} \
                   --camera-info-topics ${CAMERA_INFO_TOPICS} \
                   --robot-model 'Box()' --discard-model 'Box()' \
                   --time-step ${TIME_STEP} --visualize $VIS --save-data $SAVE \
                   --input-step 200
