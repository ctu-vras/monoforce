#!/bin/bash

MODEL_PATH=/home/$USER/workspaces/traversability_ws/src/monoforce/config/weights/monoforce/

source /home/$USER/workspaces/traversability_ws/devel/setup.bash
#./evaluate
./evaluate --model_name monolayout
#./evaluate --model_name kkt

