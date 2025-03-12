#!/bin/bash

echo "Source ROS workspace..."
source $HOME/workspaces/traversability_ws/devel/setup.bash

SEQ=val
BATCH_SIZE=1
TERRAIN_ENCODERS=(lss)
TRAJ_PREDICTORS=(dphysics)
VIS=False

for TERRAIN_ENCODER in "${TERRAIN_ENCODERS[@]}"
do
  for TRAJ_PREDICTOR in "${TRAJ_PREDICTORS[@]}"
  do
    WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/${TERRAIN_ENCODER}/val.pth
    echo "Evaluating terrain encoder ${TERRAIN_ENCODER} with trajectory predictor ${TRAJ_PREDICTOR}..."
    ./eval.py --terrain_encoder ${TERRAIN_ENCODER} \
              --terrain_encoder_path ${WEIGHTS} \
              --traj_predictor ${TRAJ_PREDICTOR} \
              --batch_size ${BATCH_SIZE} \
              --seq ${SEQ} \
              --vis ${VIS}
  done
done

echo "Done evaluating."
