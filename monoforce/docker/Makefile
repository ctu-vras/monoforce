all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build              - build image"
	@echo "   2. make clean              - remove image"
	@echo "   3. make pull               - pull image"
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
	@docker run -it --rm --net=host --runtime=nvidia --gpus all \
            -v ${HOME}/.Xauthority:/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e DISPLAY=${DISPLAY} -e XAUTHORITY=/root/.Xauthority \
            -v ${HOME}/workspaces/traversability_ws/src:/root/traversability_ws/src \
            -v /media/ruslan/SSD/data:/media/ruslan/SSD/data \
            -v ${HOME}/.cache:/root/.cache \
            agishrus/monoforce /bin/bash
