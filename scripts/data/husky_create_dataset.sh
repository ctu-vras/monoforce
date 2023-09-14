#!/bin/bash

# This script creates a dataset from a bag file
# It saves point clouds and trajectories from a bag file
# It also adds calibrations, colors and velocities to the point clouds
# It is used to create a dataset for the HUSKY robot

DATA_PATH=/home/$USER/data/robingas/data
ROBOT_NAME='husky'
#EXPERIMENT_NAME='22-09-27-unhost'/${ROBOT_NAME}
EXPERIMENT_NAME='22-10-27-unhost-final-demo'
#SEQUENCE_NAME='husky_2022-09-27-15-01-44'
#SEQUENCE_NAME='husky_2022-09-27-10-33-15'
SEQUENCE_NAME='husky_2022-10-27-15-33-57'
INPUT_DATA_STEP=20
VIS=False
SAVE=True

source /home/$USER/workspaces/traversability_ws/devel/setup.bash

./save_clouds_and_trajectories_from_bag --bag-paths $DATA_PATH/$EXPERIMENT_NAME/${SEQUENCE_NAME}.bag \
	                                            $DATA_PATH/$EXPERIMENT_NAME/${SEQUENCE_NAME}_loc.bag \
                                                    --cloud-topics /points \
                                                    --imu-topics /imu/data \
                                                    --robot-model 'Box()' --discard-model 'Box()' \
                                                    --input-step $INPUT_DATA_STEP --visualize $VIS --save-data $SAVE

./add_calibrations_colors --bag-path $DATA_PATH/$EXPERIMENT_NAME/${SEQUENCE_NAME}.bag \
                                     --lidar-topic /points \
                                     --camera-topics /camera_rear/image_color/compressed \
                                                     /camera_front/image_color/compressed \
                                                     /camera_right/image_color/compressed \
                                                     /camera_left/image_color/compressed \
                                     --camera-info-topics /camera_front/camera_info \
                                                         /camera_rear/camera_info \
                                                         /camera_right/camera_info \
                                                         /camera_left/camera_info \
                                     --visualize $VIS --save-data $SAVE

./create_terrain_features --dataset-path $DATA_PATH/$EXPERIMENT_NAME/${SEQUENCE_NAME}_trav/ \
                          --control-model diffdrive \
                          --visualize $VIS --save-data $SAVE
