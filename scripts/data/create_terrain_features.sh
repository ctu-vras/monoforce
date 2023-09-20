#! /bin/bash

DATA_PATH=/home/$USER/data/robingas/data
ROBOT_NAME='marv'
EXPERIMENT_NAME='22-08-12-cimicky_haj'
#SEQUENCE_NAME='ugv_2022-08-12-15-18-34'
SEQUENCE_NAME='ugv_2022-08-12-16-37-03'

source /home/agishrus/workspace/traversability_ws/devel/setup.bash
./create_terrain_features --dataset-path $DATA_PATH/$EXPERIMENT_NAME/$ROBOT_NAME/${SEQUENCE_NAME}_trav/ \
                          --control-model diffdrive \
                          --visualize False --save-data True
