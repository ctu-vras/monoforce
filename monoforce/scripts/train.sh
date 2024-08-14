#!/bin/bash

ROBOT=tradr
DATASET=robingas
INPUT_TYPE=rgb
ONLY_FRONT_CAM=False
DEBUG=False
VIS=False
BSZ=64

WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/lss/lss_${DATASET}_${ROBOT}.pt

source /home/$USER/workspaces/traversability_ws/devel/setup.bash

./train --bsz $BSZ --nworkers 2 --nepochs 200 --lr 0.001 --weight_decay 1e-7 \
        --debug $DEBUG --vis $VIS \
        --geom_hm_weight 1.0 --terrain_hm_weight 100.0 --hdiff_weight 1e-6 \
        --dataset $DATASET \
        --input_type $INPUT_TYPE \
        --robot $ROBOT \
        --dphys_cfg_path ../config/dphys_cfg.yaml \
        --lss_cfg_path ../config/lss_cfg_$ROBOT.yaml \
        --only_front_cam ${ONLY_FRONT_CAM} \
        --pretrained_model_path ${WEIGHTS}
