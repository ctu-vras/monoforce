all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build              - build image"
	@echo "   2. make clean              - remove image"
	@echo "   3. make pull               - pull image"
	@echo "   4. make run                - run the container"
	@echo ""

build:
	@docker build --tag agishrus/monoforce -f ./Dockerfile .

clean:
	@docker rmi -f agishrus/monoforce

pull:
	@docker pull agishrus/monoforce

run:
	@docker run -it --rm --net=host --gpus all \
        -v /media/ruslan/SSD/data/robingas/:/root/catkin_ws/src/monoforce/data/robingas/ \
		-v $(pwd)/../:/root/catkin_ws/src/monoforce/ agishrus/monoforce /bin/bash
