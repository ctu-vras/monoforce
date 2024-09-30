#!/bin/bash

ROBOT=marv
DATASET=robingas
ONLY_FRONT_CAM=False
DEBUG=False
VIS=False
BSZ=4

WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/lss/lss_${DATASET}_${ROBOT}.pt

source $HOME/workspaces/traversability_ws/devel/setup.bash

./train --bsz $BSZ --nworkers 16 --nepochs 1000 --lr 1e-5 --weight_decay 1e-7 \
        --debug $DEBUG --vis $VIS \
        --geom_hm_weight 1.0 --terrain_hm_weight 10.0 --hdiff_weight 1e-6 --phys_weight 10.0 \
        --dataset $DATASET \
        --robot $ROBOT \
        --dphys_cfg_path ../config/dphys_cfg.yaml \
        --lss_cfg_path ../config/lss_cfg_$ROBOT.yaml \
        --only_front_cam ${ONLY_FRONT_CAM} \
        --pretrained_model_path ${WEIGHTS}
