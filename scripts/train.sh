#!/bin/bash

WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/config/tb_runs/lss_robingas_husky_oru_2024_04_25_10_38_43/lss.pt
ROBOT=husky_oru
DATASET=robingas

source /home/$USER/workspaces/traversability_ws/devel/setup.bash

./train --bsz 64 --nworkers 2 --nepochs 1000 --lr 0.001 --weight_decay 1e-7 \
        --debug False --vis False \
        --geom_hm_weight 1.0 --terrain_hm_weight 10.0 --hdiff_weight 1e-4 \
        --dataset $DATASET \
        --robot $ROBOT \
        --dphys_cfg_path ../config/dphys_cfg.yaml \
        --lss_cfg_path ../config/lss_cfg_$ROBOT.yaml \
	--only_front_hm True \
        --pretrained_model_path ${WEIGHTS}
