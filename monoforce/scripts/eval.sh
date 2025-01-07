#!/bin/bash

echo "Source ROS workspace..."
source $HOME/workspaces/traversability_ws/devel/setup.bash

ROBOT=marv  # marv, tradr, husky
TERRAIN_ENCODERS=(lss voxelnet bevfusion)
TRAJ_PREDICTORS=(dphysics traj_lstm)

for TERRAIN_ENCODER in "${TERRAIN_ENCODERS[@]}"
do
  for TRAJ_PREDICTOR in "${TRAJ_PREDICTORS[@]}"
  do
    WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/${TERRAIN_ENCODER}/val.pth
    echo "Evaluating terrain encoder ${TERRAIN_ENCODER} with trajectory predictor ${TRAJ_PREDICTOR}..."
    ./eval.py --terrain_encoder ${TERRAIN_ENCODER} \
              --terrain_encoder_path ${WEIGHTS} \
              --traj_predictor ${TRAJ_PREDICTOR} \
              --robot ${ROBOT} #--vis
  done
done

echo "Done evaluating."
