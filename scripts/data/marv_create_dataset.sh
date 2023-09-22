#!/bin/bash

# This script creates a dataset from a bag file
# It saves point clouds and trajectories from a bag file
# It also adds calibrations, colors and velocities to the point clouds
# It is used to create a dataset for the MARV robot

DATA_PATH=/home/$USER/data/robingas/data
ROBOT_NAME='marv'
EXPERIMENT_NAME='22-08-12-cimicky_haj'
#SEQUENCE_NAME='ugv_2022-08-12-15-18-34'
SEQUENCE_NAME='ugv_2022-08-12-16-37-03'
INPUT_DATA_STEP=20
VIS=False
SAVE=True

source /home/$USER/workspaces/traversability_ws/devel/setup.bash

./save_clouds_and_trajectories_from_bag --bag-paths ${DATA_PATH}/${EXPERIMENT_NAME}/${ROBOT_NAME}/${SEQUENCE_NAME}.bag \
                                                    ${DATA_PATH}/${EXPERIMENT_NAME}/${ROBOT_NAME}/${SEQUENCE_NAME}_loc.bag \
                                                    --cloud-topics /os_cloud_node/destaggered_points \
                                                    --imu-topics /imu/data \
                                                    --robot-model 'Box()' --discard-model 'Box()' \
                                                    --input-step $INPUT_DATA_STEP --visualize $VIS --save-data $SAVE

./add_calibrations_colors --bag-path ${DATA_PATH}/${EXPERIMENT_NAME}/${ROBOT_NAME}/${SEQUENCE_NAME}.bag \
                                     --lidar-topic /os_cloud_node/destaggered_points \
                                     --camera-topics /camera_fisheye_rear/image_color/compressed \
                                                     /camera_fisheye_front/image_color/compressed \
                                                     /camera_right/image_color/compressed \
                                                     /camera_left/image_color/compressed \
                                                     /camera_up/image_color/compressed \
                                     --camera-info-topics /camera_fisheye_front/camera_info \
                                                         /camera_fisheye_rear/camera_info \
                                                         /camera_right/camera_info \
                                                         /camera_left/camera_info \
                                                         /camera_up/camera_info \
                                     --visualize $VIS --save-data $SAVE

./create_terrain_features --dataset-path ${DATA_PATH}/${EXPERIMENT_NAME}/${ROBOT_NAME}/${SEQUENCE_NAME}_trav/ \
                          --control-model diffdrive \
                          --visualize $VIS --save-data $SAVE
