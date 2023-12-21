#!/bin/bash

# This script creates a LSS data from a bag file
# It saves point clouds and images from a bag file

DATA_PATH=/media/ruslan/SSD/data/bags/lss_input

# list of sequences to process
BAGS=(
      ${DATA_PATH}/'husky_emptyfarm_2023-12-13-13-39-29.bag'
      ${DATA_PATH}/'husky_farmWith1CropRow_2023-12-13-14-14-14.bag'
      ${DATA_PATH}/'husky_inspection_2023-12-13-14-40-40.bag'
      ${DATA_PATH}/'husky_simcity_2023-12-13-14-18-26.bag'
      ${DATA_PATH}/'husky_simcity_dynamic_2023-12-13-14-30-09.bag'
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

# loop through bag files
for BAG in "${BAGS[@]}"
do
    echo "Processing bag file: ${BAG}"
    # save clouds and trajectories
    ./create_lss_data --bag-path ${BAG} \
                      --lidar-topic ${CLOUD_TOPIC} \
                      --camera-topics ${CAMERA_TOPICS} \
                      --camera-info-topics ${CAMERA_INFO_TOPICS}
done
