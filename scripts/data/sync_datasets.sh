#!/bin/bash

# This script synchronizes data sequences from a remote server
DATA_PATH=/home/$USER/data/bags/robingas/data

# list of sequences to process
SEQUENCES=(
            '22-09-27-unhost/husky/husky_2022-09-27-15-01-44_trav/'
            '22-09-27-unhost/husky/husky_2022-09-27-10-33-15_trav/'
            '22-10-27-unhost-final-demo/husky_2022-10-27-15-33-57_trav/'
            '22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'
            '22-08-12-cimicky_haj/marv/ugv_2022-08-12-16-37-03_trav/'
)

USER_NAME=agishrus
SERVER=login3.rci.cvut.cz

# loop through bag files
for SEQ in "${SEQUENCES[@]}"
do
    echo "Processing sequence: ${SEQ}"
    SOURCE_PATH=${USER_NAME}@$SERVER:/mnt/personal/agishrus/data/robingas/data/$SEQ
    TARGET_PATH=${DATA_PATH}/$SEQ
    echo "Synchronizing from source path ${SOURCE_PATH}"
    echo "to target path $TARGET_PATH"

    rsync -r --progress --ignore-existing --exclude='*.bag' ${SOURCE_PATH} ${TARGET_PATH}
done
echo "Done synchronizing data from remote server."
