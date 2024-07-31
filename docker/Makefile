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
	@docker build --tag agishrus/monoforce_deploy -f ./Dockerfile .

clean:
	@docker rmi -f agishrus/monoforce_deploy

pull:
	@docker pull agishrus/monoforce_deploy

shell:
	@docker run -it --rm --gpus all --net host -v /home:/home agishrus/monoforce_deploy /bin/bash