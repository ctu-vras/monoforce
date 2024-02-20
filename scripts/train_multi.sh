#!/bin/bash

EXPERIMENT_ID=$1
# if experiment ID is not provided, set it to 0
if [ -z "$EXPERIMENT_ID" ]
then
    EXPERIMENT_ID=0
fi
echo "Experiment ID: $EXPERIMENT_ID"

# source ROS workspace
WS_PATH=/home/$USER/workspaces/traversability_ws
echo "Sourcing ROS workspace: $WS_PATH"
# if ROS workspace does not exist, exit
if [ ! -d "$WS_PATH" ]
then
    echo "ROS workspace does not exist: $WS_PATH"
    exit 1
fi
# source ROS workspace
source $WS_PATH/devel/setup.bash
# current date
DATE=$(date +'%Y_%m_%d_%H_%M_%S')

case $EXPERIMENT_ID in
    0)
        echo "Training with default parameters"
        ./train --bsz 24 --nworkers 12 --nepochs 1000 --lr 0.001 --weight_decay 1e-7 \
                --map_consistency False \
                --debug False --vis False \
                --lidar_hm_weight 1.0 --traj_hm_weight 100.0 --hdiff_weight 1e-4
        ;;
    1)
        ./train --bsz 24 --nworkers 12 --nepochs 500 --lr 0.001 --weight_decay 1e-7 \
                --map_consistency False \
                --debug True --vis False \
                --lidar_hm_weight 1.0 --traj_hm_weight 10.0 --hdiff_weight 1e-4 \
                --log_dir $WS_PATH/src/monoforce/config/tb_runs/lss_$DATE/job_${EXPERIMENT_ID}
        ;;
    2)
        ./train --bsz 24 --nworkers 12 --nepochs 500 --lr 0.001 --weight_decay 1e-7 \
                --map_consistency False \
                --debug True --vis False \
                --lidar_hm_weight 1.0 --traj_hm_weight 10.0 --hdiff_weight 1e-3 \
                --log_dir $WS_PATH/src/monoforce/config/tb_runs/lss_$DATE/job_${EXPERIMENT_ID}
        ;;
    3)
        ./train --bsz 24 --nworkers 12 --nepochs 500 --lr 0.001 --weight_decay 1e-7 \
                --map_consistency False \
                --debug True --vis False \
                --lidar_hm_weight 1.0 --traj_hm_weight 10.0 --hdiff_weight 1e-2 \
                --log_dir $WS_PATH/src/monoforce/config/tb_runs/lss_$DATE/job_${EXPERIMENT_ID}
        ;;
    4)
        ./train --bsz 24 --nworkers 12 --nepochs 500 --lr 0.001 --weight_decay 1e-7 \
                --map_consistency False \
                --debug True --vis False \
                --lidar_hm_weight 1.0 --traj_hm_weight 50.0 --hdiff_weight 1e-4 \
                --log_dir $WS_PATH/src/monoforce/config/tb_runs/lss_$DATE/job_${EXPERIMENT_ID}
        ;;
    5)
        ./train --bsz 24 --nworkers 12 --nepochs 500 --lr 0.001 --weight_decay 1e-7 \
                --map_consistency False \
                --debug True --vis False \
                --lidar_hm_weight 1.0 --traj_hm_weight 50.0 --hdiff_weight 1e-3 \
                --log_dir $WS_PATH/src/monoforce/config/tb_runs/lss_$DATE/job_${EXPERIMENT_ID}
        ;;
    6)
        ./train --bsz 24 --nworkers 12 --nepochs 500 --lr 0.001 --weight_decay 1e-7 \
                --map_consistency False \
                --debug True --vis False \
                --lidar_hm_weight 1.0 --traj_hm_weight 50.0 --hdiff_weight 1e-2 \
                --log_dir $WS_PATH/src/monoforce/config/tb_runs/lss_$DATE/job_${EXPERIMENT_ID}
        ;;
    7)
        ./train --bsz 24 --nworkers 12 --nepochs 500 --lr 0.001 --weight_decay 1e-7 \
                --map_consistency False \
                --debug True --vis False \
                --lidar_hm_weight 1.0 --traj_hm_weight 100.0 --hdiff_weight 1e-4 \
                --log_dir $WS_PATH/src/monoforce/config/tb_runs/lss_$DATE/job_${EXPERIMENT_ID}
        ;;
    8)
        ./train --bsz 24 --nworkers 12 --nepochs 500 --lr 0.001 --weight_decay 1e-7 \
                --map_consistency False \
                --debug True --vis False \
                --lidar_hm_weight 1.0 --traj_hm_weight 100.0 --hdiff_weight 1e-3 \
                --log_dir $WS_PATH/src/monoforce/config/tb_runs/lss_$DATE/job_${EXPERIMENT_ID}
        ;;
    9)
        ./train --bsz 24 --nworkers 12 --nepochs 500 --lr 0.001 --weight_decay 1e-7 \
                --map_consistency False \
                --debug True --vis False \
                --lidar_hm_weight 1.0 --traj_hm_weight 100.0 --hdiff_weight 1e-2 \
                --log_dir $WS_PATH/src/monoforce/config/tb_runs/lss_$DATE/job_${EXPERIMENT_ID}
        ;;
    *)
        echo "Invalid experiment ID: $EXPERIMENT_ID"
        exit 1
        ;;
esac
echo "Done training."

