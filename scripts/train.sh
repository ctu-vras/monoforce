#!/bin/bash

source /home/$USER/workspaces/traversability_ws/devel/setup.bash
./train --bsz 24 --nworkers 12 --nepochs 1000 --lr 0.001 --weight_decay 1e-7 \
        --map_consistency False \
        --debug False --vis False \
        --lidar_hm_weight 1.0 --traj_hm_weight 100.0 --hdiff_weight 1e-4
