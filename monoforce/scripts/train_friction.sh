#!/bin/bash

ROBOT=tradr
DATASET=robingas
DEBUG=False
VIS=False
BSZ=128

WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/lss/lss_${DATASET}_${ROBOT}.pt

source $HOME/workspaces/traversability_ws/devel/setup.bash

./train_friction --bsz $BSZ --nepochs 1000 --lr 1e-5 \
                 --debug $DEBUG --vis $VIS \
                 --dataset $DATASET \
                 --robot $ROBOT \
                 --lss_cfg_path ../config/lss_cfg_$ROBOT.yaml \
                 --pretrained_model_path ${WEIGHTS}
