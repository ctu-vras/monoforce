#!/bin/bash

SEQ=val
BATCH_SIZE=1
TERRAIN_ENCODERS=(lss)
VIS=False

for TERRAIN_ENCODER in "${TERRAIN_ENCODERS[@]}"
do
  WEIGHTS=$HOME/workspaces/ros1/traversability_ws/src/monoforce/monoforce/config/weights/${TERRAIN_ENCODER}/val.pth
  echo "Evaluating terrain encoder ${TERRAIN_ENCODER}..."
  ./eval.py --terrain_encoder ${TERRAIN_ENCODER} \
            --terrain_encoder_path ${WEIGHTS} \
            --batch_size ${BATCH_SIZE} \
            --seq ${SEQ} \
            --vis ${VIS}
done

echo "Done evaluating."
