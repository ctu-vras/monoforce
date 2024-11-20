#!/bin/bash

MODEL=lss  # lss, lidarbev, bevfusion
ROBOT=tradr
DEBUG=False
VIS=False
BSZ=32  # 32, 32, 8
WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/lss/lss.pt

source $HOME/workspaces/traversability_ws/devel/setup.bash
./train.py --bsz $BSZ --nepochs 1000 --lr 1e-4 --weight_decay 1e-7 \
           --debug $DEBUG --vis $VIS \
           --terrain_weight 1.0 --phys_weight 0.01 \
           --robot $ROBOT \
           --model $MODEL \
           --pretrained_model_path ${WEIGHTS}
