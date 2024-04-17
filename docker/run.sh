#!/bin/bash
trap : SIGTERM SIGINT

function abspath() {
    # generate absolute path from relative path
    # $1     : relative filename
    # return : absolute path
    if [ -d "$1" ]; then
        # dir
        (cd "$1"; pwd)
    elif [ -f "$1" ]; then
        # file
        if [[ $1 = /* ]]; then
            echo "$1"
        elif [[ $1 == */* ]]; then
            echo "$(cd "${1%/*}"; pwd)/${1##*/}"
        else
            echo "$(pwd)/$1"
        fi
    fi
}

roscore &
ROSCORE_PID=$!
sleep 1

rviz -d ../config/rviz/monoforce_husky.rviz &
RVIZ_PID=$!

MONOFORCE_DIR=$(abspath "..")
DATA_DIR=$(abspath "../data/robingas/")

docker run \
  -it \
  --rm \
  --gpus all \
  --net=host \
  -v ${MONOFORCE_DIR}:/root/catkin_ws/src/monoforce/ \
  -v ${DATA_DIR}:/root/catkin_ws/src/monoforce/data/robingas/ \
  agishrus/monoforce \
  /bin/bash -c \
  "cd /root/catkin_ws/; \
  catkin config \
      --cmake-args \
          -DCMAKE_BUILD_TYPE=Release; \
      catkin build; \
      source devel/setup.bash; \
      roslaunch monoforce monoforce.launch rviz:=false robot:=husky"

wait $ROSCORE_PID
wait $RVIZ_PID

if [[ $? -gt 128 ]]
then
    kill $ROSCORE_PID
    kill $RVIZ_PID
fi
