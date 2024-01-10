#!/bin/bash

# This script creates a LSS data from a bag file
# It saves point clouds and images from a bag file

DATA_PATH=/home/ruslan/data/bags/lss_input
#DATA_PATH=/media/ruslan/SSD/data/bags/lss_input

# list of sequences to process
BAGS=(
      ${DATA_PATH}/'husky_emptyfarm_2024-01-03-13-36-25.bag'
      ${DATA_PATH}/'husky_farmWith1CropRow_2024-01-03-13-52-36.bag'
      ${DATA_PATH}/'husky_inspection_2024-01-03-14-06-53.bag'
      ${DATA_PATH}/'husky_simcity_2024-01-03-13-55-37.bag'
      ${DATA_PATH}/'husky_simcity_dynamic_2024-01-03-13-59-08.bag'
      ${DATA_PATH}/'husky_simcity_2024-01-09-17-56-34.bag'
      ${DATA_PATH}/'husky_simcity_2024-01-09-17-50-23.bag'
      ${DATA_PATH}/'husky_emptyfarm_vegetation_2024-01-09-17-18-46.bag'
)

# source ROS workspace
source /home/$USER/workspaces/traversability_ws/devel/setup.bash

CLOUD_TOPIC='/points'
CAMERA_TOPICS='/realsense_front/color/image_raw/compressed
               /realsense_left/color/image_raw/compressed
               /realsense_rear/color/image_raw/compressed
               /realsense_right/color/image_raw/compressed'
CAMERA_INFO_TOPICS='/realsense_front/color/camera_info
                    /realsense_left/color/camera_info
                    /realsense_rear/color/camera_info
                    /realsense_right/color/camera_info'
DEPTH_TOPICS='/realsense_front/depth/image_rect_raw
              /realsense_left/depth/image_rect_raw
              /realsense_rear/depth/image_rect_raw
              /realsense_right/depth/image_rect_raw'

# loop through bag files
for BAG in "${BAGS[@]}"
do
    echo "Processing bag file: ${BAG}"
    # save clouds and trajectories
    ./create_lss_data --bag-path ${BAG} \
                      --lidar-topic ${CLOUD_TOPIC} \
                      --camera-topics ${CAMERA_TOPICS} \
                      --camera-info-topics ${CAMERA_INFO_TOPICS} \
                      --depth-topics ${DEPTH_TOPICS} \
                      --time-period 1.0
done
