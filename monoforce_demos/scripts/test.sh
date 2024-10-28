#!/bin/bash

WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/lss/lss.pt
ROBOT=tradr2
SEQ_I=0

source $HOME/workspaces/traversability_ws/devel/setup.bash
./test.py --model_path ${WEIGHTS} --robot ${ROBOT} --seq_i ${SEQ_I}
