#!/bin/bash

MODEL=lss
DEBUG=False
VIS=False
BSZ=32
WEIGHTS=$HOME/workspaces/ros2/traversability_ws/src/monoforce/monoforce/config/weights/${MODEL}/val.pth

source $HOME/workspaces/ros2/traversability_ws/install/setup.bash
python train.py --batch_size $BSZ --n_epochs 1000 --lr 1e-3 \
                --debug $DEBUG --vis $VIS \
                --geom_weight 1.0 --terrain_weight 0.0 --phys_weight 0.0 --logvar_weight 2.0 \
                --pretrained_terrain_encoder_path ${WEIGHTS}
