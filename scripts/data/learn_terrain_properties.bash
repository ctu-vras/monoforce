#!/bin/bash

VIS=True
SAVE=False

DATA_PATH=/media/ruslan/SSD/data

# list of sequences to process
SEQUENCES=(
            'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-15-01-44/'
            'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-10-33-15/'
            'robingas/data/22-10-27-unhost-final-demo/husky_2022-10-27-15-33-57/'
            'robingas/data/22-09-23-unhost/husky/husky_2022-09-23-12-38-31/'
            'robingas/data/22-06-30-cimicky_haj/husky_2022-06-30-15-58-37/'
)


# source ROS workspace
source /home/$USER/workspaces/traversability_ws/devel/setup.bash

# learn and save terrain properties
for SEQ in "${SEQUENCES[@]}"
do
    ./learn_terrain_properties --data-path ${DATA_PATH}/$SEQ \
                               --visualize $VIS --save-data $SAVE
done
