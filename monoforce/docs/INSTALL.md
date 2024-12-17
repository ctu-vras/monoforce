# Installation

Install Python dependencies:
```commandline
pip install -r ../singularity/requirements.txt
```

Please clone the ROS package, install its dependencies, and build the workspace:
```commandline
mkdir -p ~/traversability_ws/src/
cd ~/traversability_ws/src/
git clone https://github.com/ctu-vras/monoforce.git

wstool init
wstool merge monoforce/monoforce_gazebo/dependencies.rosinstall
wstool merge monoforce/monoforce_navigation/dependencies.rosinstall
wstool up -j 4

cd ~/traversability_ws/
catkin init
catkin config --extend /opt/ros/$ROS_DISTRO/
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
rosdep install --from-paths src --ignore-src -r -y
catkin build
```

### Model Weights

The pretrained weights for the terrain encoder are available at:
[https://github.com/ctu-vras/monoforce/releases/download/v0.1.1/weights.zip](https://github.com/ctu-vras/monoforce/releases/download/v0.1.1/weights.zip).

Once downloaded, please, unzip the weights to the
`monoforce/config/weights/` folder.

## Docker

We have prepared a [Docker](https://docs.docker.com/engine/install/ubuntu/) image to run the monoforce package.
Please, install
[Docker](https://docs.docker.com/engine/install/ubuntu/)
and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

To pull the prebuilt image:
```commandline
cd ../docker/
make pull
```

Or if you would like to build the image from scratch:
```commandline
cd ../docker/
make build
```

## Singularity

It is possible to run the monoforce package in a [Singularity](https://sylabs.io/singularity/) container.
To build the image:
```commandline
cd ../singularity/
./build.sh
```
