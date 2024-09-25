#!/bin/bash

ROBOT=tradr
DATASET=robingas
DEBUG=True
VIS=False
BSZ=4

WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/lss/lss_${DATASET}_${ROBOT}.pt

source $HOME/workspaces/traversability_ws/devel/setup.bash

./train_friction --bsz $BSZ --nepochs 1000 --lr 1e-5 \
                 --debug $DEBUG --vis $VIS \
                 --dataset $DATASET \
                 --robot $ROBOT \
                 --dphys_cfg_path ../config/dphys_cfg.yaml \
                 --lss_cfg_path ../config/lss_cfg_$ROBOT.yaml \
                 --pretrained_model_path ${WEIGHTS}
