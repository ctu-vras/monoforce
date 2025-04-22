#!/bin/bash

echo "Source ROS workspace..."
source $HOME/workspaces/ros2/traversability_ws/install/setup.bash

BATCH_SIZE=1
TERRAIN_ENCODER=lss
VIS=True

WEIGHTS=$HOME/workspaces/ros2/traversability_ws/src/monoforce/monoforce/config/weights/${TERRAIN_ENCODER}/val.pth
echo "Evaluating terrain encoder ${TERRAIN_ENCODER}..."
python eval.py --pretrained_terrain_encoder_path ${WEIGHTS} \
               --batch_size ${BATCH_SIZE} \
               --vis ${VIS}
echo "Done evaluating."