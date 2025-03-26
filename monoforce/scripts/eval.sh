#!/bin/bash

echo "Source ROS workspace..."
source $HOME/workspaces/traversability_ws/devel/setup.bash

SEQ=val
BATCH_SIZE=1
TERRAIN_ENCODERS=(lss)
VIS=True

for TERRAIN_ENCODER in "${TERRAIN_ENCODERS[@]}"
do
  WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/${TERRAIN_ENCODER}/val.pth
  echo "Evaluating terrain encoder ${TERRAIN_ENCODER} with trajectory predictor ${TRAJ_PREDICTOR}..."
  python eval.py --terrain_encoder ${TERRAIN_ENCODER} \
                 --terrain_encoder_path ${WEIGHTS} \
                 --batch_size ${BATCH_SIZE} \
                 --seq ${SEQ} \
                 --vis ${VIS}
done

echo "Done evaluating."