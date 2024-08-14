# Installation

Install Python dependencies:
```commandline
pip install -r ../singularity/requirements.txt
```

Please clone the ROS package, install its dependencies, and build the workspace:
```commandline
mkdir -p ~/catkin_ws/src/
cd ~/catkin_ws/src/
git clone https://github.com/ctu-vras/monoforce.git

wstool init
wstool merge monoforce/monoforce_gazebo/dependencies.rosinstall
wstool merge monoforce/monoforce_navigation/dependencies.rosinstall
wstool up -j 4

cd ~/catkin_ws/
catkin init
catkin config --extend /opt/ros/$ROS_DISTRO/
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
rosdep install --from-paths src --ignore-src -r -y
catkin build
```

### Model Weights

The pretrained weights for the LSS terrain encoder can be downloaded from:
- RobinGas: [lss_robingas_husky.pt](https://drive.google.com/file/d/1h1VieiIdGZB1Ml3QdIlh8ZJA67sJej4m/view?usp=sharing),
            [lss_robingas_tradr.pt](https://drive.google.com/file/d/1jpsgXN-44Bbu9hfAWd5Z3te1DWp3s8cX/view?usp=sharing),
            [lss_robingas_husky_oru.pt](https://drive.google.com/file/d/12v6EAvaw0LqdINYFyHYr0t5mlZn-VN6c/view?usp=sharing),
- RELLIS-3D: [lss_rellis3d.pt](https://drive.google.com/file/d/1kK75mUxHn-4GadU4k8-c43hA9t3bZxw1/view?usp=sharing).

Once downloaded put the weights to `monoforce/config/weights/lss` folder.

## Docker

We have prepared a [Docker](https://docs.docker.com/engine/install/ubuntu/) image to run the monoforce package.

To pull the image:
```commandline
cd ../docker/
make pull
```

To build the image from scratch:
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
