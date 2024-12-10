#!/bin/bash

MODEL=lss  # lss, lidarbev, bevfusion
ROBOT=marv
DEBUG=False
VIS=False
BSZ=32  # 32, 32, 8
WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/${MODEL}/val.pth

source $HOME/workspaces/traversability_ws/devel/setup.bash
./train.py --bsz $BSZ --nepochs 1000 --lr 1e-3 --weight_decay 1e-7 \
           --debug $DEBUG --vis $VIS \
           --geom_weight 1.0 --terrain_weight 2.0 --phys_weight 1.0 \
           --traj_sim_time 5.0 \
           --robot $ROBOT \
           --model $MODEL \
           --pretrained_model_path ${WEIGHTS}
