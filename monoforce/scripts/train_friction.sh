#!/bin/bash

ROBOT=tradr2
DATASET=rough
DEBUG=False
VIS=False
BSZ=128

WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/lss/lss.pt

source $HOME/workspaces/traversability_ws/devel/setup.bash

./train_friction --bsz $BSZ --nepochs 1000 --lr 1e-4 \
                 --debug $DEBUG --vis $VIS \
                 --dataset $DATASET \
                 --robot $ROBOT \
                 --pretrained_model_path ${WEIGHTS}
