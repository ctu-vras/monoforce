#!/bin/bash

source /home/$USER/workspaces/traversability_ws/devel/setup.bash
./train_monoforce --batch_size 64 --regularization 0.2

