#!/bin/bash

WS_PATH=/home/$USER/workspaces/traversability_ws
ROBOT=tradr

source $WS_PATH/devel/setup.bash

./eval  --vis False \
        --val_fraction 0.1 \
        --robot $ROBOT \
        --dataset robingas \
        --dphys_config ../config/dphys_cfg.yaml \
	      --lss_config ../config/lss_cfg_$ROBOT.yaml
