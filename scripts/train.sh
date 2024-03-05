#!/bin/bash

source /home/$USER/workspaces/traversability_ws/devel/setup.bash

WEIGHTS=/home/agishrus/workspaces/traversability_ws/src/monoforce/config/tb_runs/lss_2024_03_02_18_21_07/train_lss.pt
./train --bsz 96 --nworkers 12 --nepochs 1000 --lr 0.001 --weight_decay 1e-7 \
        --map_consistency False \
        --debug False --vis False \
        --lidar_hm_weight 1.0 --traj_hm_weight 100.0 --hdiff_weight 0.005 \
	--pretrained_model_path ${WEIGHTS}

