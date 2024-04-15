#!/bin/bash

source /home/$USER/workspaces/traversability_ws/devel/setup.bash

WEIGHTS=/home/agishrus/workspaces/traversability_ws/src/monoforce/config/tb_runs/lss_robingas_2024_04_13_19_26_27/lss.pt

./train --bsz 32 --nworkers 12 --nepochs 1000 --lr 0.001 --weight_decay 1e-7 \
        --debug False --vis False \
        --geom_hm_weight 1.0 --terrain_hm_weight 10.0 --hdiff_weight 1e-4 \
        --dataset robingas \
	--pretrained_model_path ${WEIGHTS}

