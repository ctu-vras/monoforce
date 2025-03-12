all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build              - build image"
	@echo "   2. make clean              - remove image"
	@echo "   3. make pull               - pull image"
	@echo "   4. make push               - push image"
	@echo "   5. make shell              - run container"
	@echo "   6. make help               - show this menu"
	@echo ""

build:
	@docker build --tag agishrus/monoforce -f ./Dockerfile .

clean:
	@docker rmi -f agishrus/monoforce

pull:
	@docker pull agishrus/monoforce

push:
	@docker push agishrus/monoforce

shell:
	@docker run -it --privileged --rm --net=host --runtime=nvidia --gpus all \
            -v ${HOME}/.Xauthority:/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=${DISPLAY} -e XAUTHORITY=/root/.Xauthority \
            -v ${HOME}/workspaces/traversability_ws/src:/root/workspaces/traversability_ws/src \
            -v /media/${USER}:/media/${USER} \
            -v ${HOME}/.cache:/root/.cache \
            -v ${HOME}/.gazebo/models:/root/.gazebo/models \
            agishrus/monoforce /bin/bash
