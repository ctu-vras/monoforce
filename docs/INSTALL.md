# Installation

Prerequisites:
- [pytorch](https://pytorch.org/) for differentiable physics simulation and models training.
- [mayavi](https://docs.enthought.com/mayavi/mayavi/) for data visualization.
- [ROS](http://wiki.ros.org/ROS/Installation) for data processing, and package integration.

Install Python dependencies:
```commandline
pip install -r singularity/requirements.txt
```

Please clone and build the ROS package:
```commandline
mkdir -p ~/catkin_ws/src/
cd ~/catkin_ws/src/ && git clone https://github.com/ctu-vras/monoforce.git
cd ~/catkin_ws/ && catkin_make
```

## Singularity

It is possible to run the monoforce package in a [Singularity](https://sylabs.io/singularity/) container.
Build the container:
```commandline
cd ./singularity/
./build.sh
```
