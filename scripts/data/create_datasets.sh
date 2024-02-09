#!/bin/bash

# This script creates a dataset from a bag file
# It saves point clouds and trajectories from a bag file
# It also adds calibrations, colors and velocities to the point clouds

DATA_PATH=/home/$USER/data/robingas/data
if [ ! -d "$DATA_PATH" ]; then
    echo "Data path does not exist: $DATA_PATH"
    exit 1
fi
INPUT_DATA_STEP=20
IMU_TOPIC='/imu/data'
CONTROL_MODEL='diffdrive'
N_TERRAIN_OPT_ITERS=200
VIS=False
SAVE=True

# list of sequences to process
SEQUENCES=(
            ${DATA_PATH}/'22-09-27-unhost/husky/husky_2022-09-27-15-01-44'
            ${DATA_PATH}/'22-09-27-unhost/husky/husky_2022-09-27-10-33-15'
            ${DATA_PATH}/'22-10-27-unhost-final-demo/husky_2022-10-27-15-33-57'
            ${DATA_PATH}/'22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34'
            ${DATA_PATH}/'22-08-12-cimicky_haj/marv/ugv_2022-08-12-16-37-03'
)

# source ROS workspace
if [ ! -d "/home/$USER/workspaces/traversability_ws" ]; then
    echo "ROS workspace does not exist: /home/$USER/workspaces/traversability_ws"
    exit 1
fi
source /home/$USER/workspaces/traversability_ws/devel/setup.bash

# loop through bag files
for SEQ in "${SEQUENCES[@]}"
do
    if [ ! -f "${SEQ}.bag" ]; then
        echo "Bag file does not exist: ${SEQ}.bag"
        continue
    fi
    echo "Processing bag file: ${SEQ}.bag"
    # if husky in bag file name, use cloud topic /points
    if [[ $SEQ == *"husky"* ]]; then
        CLOUD_TOPIC='/points'
        CAMERA_TOPICS='/camera_rear/image_color/compressed
                       /camera_front/image_color/compressed
                       /camera_right/image_color/compressed
                       /camera_left/image_color/compressed'
        CAMERA_INFO_TOPICS='/camera_front/camera_info
                            /camera_rear/camera_info
                            /camera_right/camera_info
                            /camera_left/camera_info'
    # if marv in bag file name, use cloud topic /os_cloud_node/destaggered_points
    elif [[ $SEQ == *"ugv"* ]]; then
        CLOUD_TOPIC='/os_cloud_node/destaggered_points'
        CAMERA_TOPICS='/camera_fisheye_rear/image_color/compressed
                       /camera_fisheye_front/image_color/compressed
                       /camera_right/image_color/compressed
                       /camera_left/image_color/compressed
                       /camera_up/image_color/compressed'
        CAMERA_INFO_TOPICS='/camera_fisheye_front/camera_info
                            /camera_fisheye_rear/camera_info
                            /camera_right/camera_info
                            /camera_left/camera_info
                            /camera_up/camera_info'
    fi

    # save clouds and trajectories
    ./save_clouds_and_trajectories_from_bag --bag-paths ${SEQ}.bag ${SEQ}_loc.bag \
                                                        --cloud-topics ${CLOUD_TOPIC} \
                                                        --imu-topics ${IMU_TOPIC} \
                                                        --robot-model 'Box()' --discard-model 'Box()' \
                                                        --input-step $INPUT_DATA_STEP --visualize $VIS --save-data $SAVE
    # add calibrations, colors and velocities
    ./add_calibrations_colors --bag-path ${SEQ}.bag \
                                          --lidar-topic ${CLOUD_TOPIC} \
                                          --camera-topics ${CAMERA_TOPICS} \
                                          --camera-info-topics ${CAMERA_INFO_TOPICS} \
                                          --visualize $VIS --save-data $SAVE
    # create terrain features
    ./create_terrain_features --dataset-path ${SEQ}_trav/ \
                              --control-model ${CONTROL_MODEL} \
                              --n-train-iters ${N_TERRAIN_OPT_ITERS} \
                              --visualize $VIS --save-data $SAVE
done
