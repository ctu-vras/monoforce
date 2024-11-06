#!/bin/bash

ROBOT=marv
ONLY_FRONT_CAM=False
DEBUG=True
VIS=False
BSZ=4
WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/lss/lss.pt

source $HOME/workspaces/traversability_ws/devel/setup.bash
./train.py --bsz $BSZ --nepochs 1000 --lr 1e-3 --weight_decay 1e-7 \
           --debug $DEBUG --vis $VIS \
           --geom_hm_weight 1.0 --terrain_hm_weight 100.0 --hdiff_weight 1e-6 --phys_weight 0.1 \
           --robot $ROBOT \
           --only_front_cam ${ONLY_FRONT_CAM} \
           --pretrained_model_path ${WEIGHTS}
