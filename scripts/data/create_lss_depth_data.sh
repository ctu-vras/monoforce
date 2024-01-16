#!/bin/bash

# This script creates a LSS data from a bag file
# It saves point clouds and images from a bag file

DATA_PATH=/home/ruslan/data/bags/husky_sim/depth
#DATA_PATH=/media/ruslan/SSD/data/bags/husky_sim/depth

# list of sequences to process
BAGS=(
#      ${DATA_PATH}/'husky_simcity_2024-01-16-14-52-02_depth.bag'
      ${DATA_PATH}/'husky_simcity_2024-01-16-11-23-56_depth.bag'
)

# source ROS workspace
source /home/$USER/workspaces/traversability_ws/devel/setup.bash

DEPTH_TOPICS='/realsense_front/depth/image_rect_raw
              /realsense_left/depth/image_rect_raw
              /realsense_rear/depth/image_rect_raw
              /realsense_right/depth/image_rect_raw'
DEPTH_CAMERA_INFO_TOPICS='/realsense_front/depth/camera_info
                          /realsense_left/depth/camera_info
                          /realsense_rear/depth/camera_info
                          /realsense_right/depth/camera_info'
DEPTH_CLOUD_TOPICS='/realsense_front/depth/color/points
                    /realsense_left/depth/color/points
                    /realsense_rear/depth/color/points
                    /realsense_right/depth/color/points'

# loop through bag files
for BAG in "${BAGS[@]}"
do
    echo "Processing bag file: ${BAG}"
    # save clouds and trajectories
    ./create_lss_depth_data --bag-path ${BAG} \
                            --depth-topics ${DEPTH_TOPICS} \
                            --depth-cloud-topics ${DEPTH_CLOUD_TOPICS} \
                            --depth-camera-info-topics ${DEPTH_CAMERA_INFO_TOPICS} \
                            --time-period 1.0
done
