#!/bin/bash

ROBOT=marv
DEBUG=False
BSZ=8
#WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/lss/lss.pt

source $HOME/workspaces/traversability_ws/devel/setup.bash

./train_bevfusion.py --bsz $BSZ --nepochs 1000 --lr 1e-3 --weight_decay 1e-7 \
           --debug $DEBUG \
           --geom_hm_weight 1.0 --terrain_hm_weight 100.0 --hdiff_weight 1e-6 --phys_weight 0.0 \
           --robot $ROBOT #\
#           --pretrained_model_path ${WEIGHTS}
