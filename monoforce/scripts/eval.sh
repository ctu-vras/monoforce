#!/bin/bash

TERRAIN_ENCODER=lss  # lss, voxelnet, bevfusion
WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/${TERRAIN_ENCODER}/val.pth
ROBOT=marv

source $HOME/workspaces/traversability_ws/devel/setup.bash

echo "Evaluating model ${TERRAIN_ENCODER}..."
./eval.py --terrain_encoder ${TERRAIN_ENCODER} --terrain_encoder_path ${WEIGHTS} --robot ${ROBOT} --vis
echo "Done evaluating."
