all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build              - build all images"
	@echo "   2. make clean              - remove all images"
	@echo "   3. make run                - run the container"
	@echo ""

build:
	@docker build --tag ros:monoforce -f ./Dockerfile .

clean:
	@docker rmi -f ros:monoforce

run:
	@docker run -it --rm --net=host --gpus all \
        -v /media/ruslan/SSD/data/robingas/:/root/catkin_ws/src/monoforce/data/robingas/ \
		-v $(pwd)/../:/root/catkin_ws/src/monoforce/ ros:monoforce /bin/bash
