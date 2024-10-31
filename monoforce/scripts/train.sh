#!/bin/bash

ROBOT=marv
DATASET=rough
ONLY_FRONT_CAM=False
DEBUG=False
VIS=False
BSZ=32

WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/lss/lss.pt
#WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/tb_runs/lss_robingas_marv/2024_10_12_18_07_19/lss.pt
#WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/tb_runs/lss_robingas_tradr2/2024_10_12_18_05_02/lss.pt

source $HOME/workspaces/traversability_ws/devel/setup.bash

./train.py --bsz $BSZ --nepochs 1000 --lr 1e-3 --weight_decay 1e-7 \
           --debug $DEBUG --vis $VIS \
           --geom_hm_weight 1.0 --terrain_hm_weight 100.0 --hdiff_weight 1e-6 --phys_weight 0.1 \
           --dataset $DATASET \
           --robot $ROBOT \
           --only_front_cam ${ONLY_FRONT_CAM} \
           --pretrained_model_path ${WEIGHTS}

