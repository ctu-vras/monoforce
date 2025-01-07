#!/bin/bash

TERRAIN_ENCODER=voxelnet  # lss, voxelnet, bevfusion
TRAJ_PREDICTOR=traj_lstm  # dphysics, traj_lstm
WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/${TERRAIN_ENCODER}/val.pth
ROBOT=marv

source $HOME/workspaces/traversability_ws/devel/setup.bash

echo "Evaluating model ${TERRAIN_ENCODER}..."
./eval.py --terrain_encoder ${TERRAIN_ENCODER} \
          --terrain_encoder_path ${WEIGHTS} \
          --traj_predictor ${TRAJ_PREDICTOR} \
          --robot ${ROBOT} \
          --vis
echo "Done evaluating."
