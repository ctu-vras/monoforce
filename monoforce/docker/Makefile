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
