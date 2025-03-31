#!/bin/bash

echo "Source ROS workspace..."
source $HOME/workspaces/traversability_ws/devel/setup.bash

BATCH_SIZE=1
TERRAIN_ENCODER=lss
TRAJ_PREDICTOR=dphysics
VIS=True

WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/${TERRAIN_ENCODER}/val.pth
echo "Evaluating terrain encoder ${TERRAIN_ENCODER} with trajectory predictor ${TRAJ_PREDICTOR}..."
python eval.py --pretrained_terrain_encoder_path ${WEIGHTS} \
               --batch_size ${BATCH_SIZE} \
               --vis ${VIS}
echo "Done evaluating."