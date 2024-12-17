#!/bin/bash

MODEL=lss  # lss, lidarbev, bevfusion
WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/${MODEL}/val.pth
ROBOT=marv

source $HOME/workspaces/traversability_ws/devel/setup.bash
# loop through data sequences
for SEQ_I in {0..18};
do
    echo "Evaluating sequence ${SEQ_I}"
    ./eval.py --model_path ${WEIGHTS} --robot ${ROBOT} --seq_i ${SEQ_I} --vis #--save
done
echo "Done evaluating sequences."