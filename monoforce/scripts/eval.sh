#!/bin/bash

WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/lss/lss.pt
ROBOT=tradr

source $HOME/workspaces/traversability_ws/devel/setup.bash
# loop through data sequences
for SEQ_I in {0..27};
do
    echo "Evaluating sequence ${SEQ_I}"
    ./eval.py --model_path ${WEIGHTS} --robot ${ROBOT} --seq_i ${SEQ_I} # --vis
done
echo "Done evaluating sequences."