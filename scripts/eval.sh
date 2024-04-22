#!/bin/bash

WS_PATH=/home/$USER/workspaces/traversability_ws
MONOFORCE_PATH=$WS_PATH/src/monoforce
ROBOT=tradr

source $WS_PATH/devel/setup.bash

./eval  --debug False --vis False \
        --robot $ROBOT \
        --dataset robingas \
        --dphys_config $MONOFORCE_PATH/config/dphys_cfg.yaml \
	      --lss_config $MONOFORCE_PATH/config/lss_cfg_$ROBOT.yaml
