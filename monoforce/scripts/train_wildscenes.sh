#!/bin/bash

MODEL=lss
ROBOT=marv
DEBUG=False
VIS=False
BSZ=32

source $HOME/workspaces/traversability_ws/devel/setup.bash
./train_wildscenes.py --bsz $BSZ --nepochs 1000 --lr 1e-3 --weight_decay 1e-7 \
                      --debug $DEBUG --vis $VIS \
                      --geom_weight 1.0 --terrain_weight 2.0 \
                      --robot $ROBOT \
                      --model $MODEL
