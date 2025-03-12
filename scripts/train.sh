#!/bin/bash

MODEL=lss
ROBOT=marv
DEBUG=False
VIS=False
BSZ=24  # 24, 24, 4
WEIGHTS=$HOME/workspaces/ros2/traversability_ws/src/monoforce/config/weights/${MODEL}/val.pth

./train.py --bsz $BSZ --nepochs 1000 --lr 1e-4 \
           --debug $DEBUG --vis $VIS \
           --geom_weight 1.0 --terrain_weight 3.0 --phys_weight 4.0 \
           --traj_sim_time 5.0 \
           --robot $ROBOT \
           --model $MODEL \
           --pretrained_model_path ${WEIGHTS}
