#!/bin/bash

ROBOT=tradr
DATASET=robingas
ONLY_FRONT_CAM=False
USE_SAM_RIGID_SEMANTICS=True
DEBUG=False
VIS=False
BSZ=16

WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/lss/lss_${DATASET}_${ROBOT}.pt

source $HOME/workspaces/traversability_ws/devel/setup.bash

./train --bsz $BSZ --nworkers 2 --nepochs 1000 --lr 0.00001 --weight_decay 1e-7 \
        --debug $DEBUG --vis $VIS \
        --geom_hm_weight 1.0 --terrain_hm_weight 100.0 --hdiff_weight 1e-6 --phys_weight 1.0 \
        --dataset $DATASET \
        --robot $ROBOT \
        --dphys_cfg_path ../config/dphys_cfg.yaml \
        --lss_cfg_path ../config/lss_cfg_$ROBOT.yaml \
        --only_front_cam ${ONLY_FRONT_CAM} \
        --use_rigid_semantics ${USE_SAM_RIGID_SEMANTICS}  # \
#        --pretrained_model_path ${WEIGHTS}
