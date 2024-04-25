#!/bin/bash

WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/config/weights/lss/lss_rellis3d.pt
ROBOT=husky
DATASET=robingas

source /home/$USER/workspaces/traversability_ws/devel/setup.bash

./train --bsz 32 --nworkers 12 --nepochs 1000 --lr 0.001 --weight_decay 1e-7 \
        --debug True --vis True \
        --geom_hm_weight 1.0 --terrain_hm_weight 10.0 --hdiff_weight 1e-4 \
        --dataset $DATASET \
        --robot $ROBOT \
        --dphys_cfg_path ../config/dphys_cfg.yaml \
        --lss_cfg_path ../config/lss_cfg_$ROBOT.yaml
#	      --pretrained_model_path ${WEIGHTS}
