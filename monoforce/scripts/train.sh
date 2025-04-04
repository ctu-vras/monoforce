#!/bin/bash

MODEL=lss
DEBUG=True
VIS=True
BSZ=4
WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/tb_runs/rough/lss_2025_04_04_10_35_32/val.pth

source $HOME/workspaces/traversability_ws/devel/setup.bash
python train.py --batch_size $BSZ --n_epochs 1000 --lr 1e-4 \
                --debug $DEBUG --vis $VIS \
                --geom_weight 0.0 --terrain_weight 0.0 --phys_weight 1.0 \
                --pretrained_terrain_encoder_path ${WEIGHTS}
