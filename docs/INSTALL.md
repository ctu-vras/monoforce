# Installation

Prerequisites:
- [pytorch](https://pytorch.org/) for differentiable physics simulation and models training.
- [mayavi](https://docs.enthought.com/mayavi/mayavi/) for data visualization.
- [ROS](http://wiki.ros.org/ROS/Installation) for data processing, and package integration.

Install Python dependencies:
```commandline
pip install -r singularity/requirements.txt
```

Please clone the ROS package, install its dependencies, and build the workspace:
```commandline
mkdir -p ~/catkin_ws/src/
cd ~/catkin_ws/src/ && git clone https://github.com/ctu-vras/monoforce.git
cd ~/catkin_ws/ && rosdep install --from-paths src --ignore-src -r -y
cd ~/catkin_ws/ && catkin_make
```

## Docker

We have prepared a [Docker](https://docs.docker.com/engine/install/ubuntu/) image to run the monoforce package.
To build the image:
```commandline
cd ./docker/
make build
```

## Singularity

It is possible to run the monoforce package in a [Singularity](https://sylabs.io/singularity/) container.
To build the image:
```commandline
cd ./singularity/
./build.sh
```
