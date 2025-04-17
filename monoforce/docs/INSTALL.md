# Installation

If you would like to use the package without ROS, please install its Python dependencies:
```commandline
cd /path/to/monoforce
pip install -r monoforce/config/singularity/requirements.txt
```

## Model Weights

The pretrained weights for the terrain encoder are available at:
[https://github.com/ctu-vras/monoforce/releases/download/v0.2.0/weights.zip](https://github.com/ctu-vras/monoforce/releases/download/v0.2.0/weights.zip).

Once downloaded, please, unzip the weights to the
`/path/to/monoforce/monoforce/config/weights/` folder.

## ROS2 Integration

Please clone the package, install its dependencies, and build the workspace:
```commandline
mkdir -p ~/traversability_ws/src/
cd ~/traversability_ws/src/
git clone https://github.com/ctu-vras/monoforce.git

cd ~/traversability_ws/
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select monoforce
```
