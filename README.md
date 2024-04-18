# MonoForce

[![Arxiv](http://img.shields.io/badge/paper-arxiv.2303.01123-critical.svg?style=plastic)](https://arxiv.org/abs/2309.09007)
[![Slides](http://img.shields.io/badge/presentation-slides-orange.svg?style=plastic)](https://docs.google.com/presentation/d/1pJFHBYVeOULi-w19_mLEbDTqvvk6klcVrrYc796-2Hw/edit?usp=sharing)
[![Video](http://img.shields.io/badge/video-1min-blue.svg?style=plastic)](https://drive.google.com/file/d/1tTt1Oi5k1jKPDYn3CnzArhV3NPSNxKvD/view?usp=sharing)

Robot-terrain interaction prediction from only RGB images as input.

![](./docs/imgs/monoforce.gif)

## Table of Contents
- [Installation Instructions](./docs/INSTALL.md)
- [Traversability Data Structure and Processing (RobinGas)](./docs/DATA.md)
- [Terrain Encoder](./docs/TERRAIN_ENCODER.md)
- [Differentiable Physics](./docs/DPHYS.md)
- [Running](#running)
- [ROS Integration](#ros-integration)
- [Citation](#citation)

## Running

The MonoForce pipeline consists of the Terrain Encoder and the Differentiable Physics modules.
Given input RGB images and cameras calibration the Terrain Encoder predicts robot's supporting terrain.
Then the Differentiable Physics module simulates robot trajectory and interaction forces on the predicted terrain
for a provided control sequence (linear and angular velocities).

Please run the following command to explore the MonoForce pipeline:
```commandline
python scripts/run --img-paths IMG1_PATH IMG2_PATH ... IMGN_PATH --cameras CAM1 CAM2 ... CAMN --calibration-path CALIB_PATH
```

For example if you want to test the model with the provided images from the RobinGas dataset:
```commandline
python scripts/run --img-paths data/robingas/data/22-10-20-unhost/ugv_2022-10-20-13-58-22/images/1666267292_537972927_camera_front.png \
                               data/robingas/data/22-10-20-unhost/ugv_2022-10-20-13-58-22/images/1666267292_537972927_camera_left.png \
                               data/robingas/data/22-10-20-unhost/ugv_2022-10-20-13-58-22/images/1666267292_537972927_camera_right.png \
                   --cameras camera_front camera_left camera_right \
                   --calibration-path data/robingas/data/22-10-20-unhost/ugv_2022-10-20-13-58-22/calibration/ \
                   --lss_cfg_path config/lss_cfg_tradr.yaml --model_path config/weights/lss/lss_tradr.pt --dphys_cfg_path config/dphys_cfg.yaml \
                   --linear-vel 0.5 --angular-vel -0.1
```

If you have [ROS](http://wiki.ros.org/noetic/Installation/Ubuntu) and [Docker](https://docs.docker.com/engine/install/ubuntu/) installed you can also run:
```commandline
cd docker/ && ./run.sh
```

<img src="./docs/imgs/tradr_rgb_input.png" width="800"/>
<img src="./docs/imgs/monoforce_mayavi.gif" width="800"/>

## ROS Integration

We provide a ROS node that integrates the trained Terrain Encoder model with the Differentiable Physics module.
Given the input RGB images and cameras calibration, the Terrain Encoder predicts the terrain shape,
which is then used to simulate robot trajectories.

```commandline
roslaunch monoforce monoforce.launch
```

## Citation

Consider citing the paper if you find it relevant to your research:

```bibtex
@article{agishev2023monoforce,
    title={MonoForce: Self-supervised Learning of Physics-aware Model for Predicting Robot-terrain Interaction},
    author={Ruslan Agishev and Karel Zimmermann and Vladimír Kubelka and Martin Pecka and Tomáš Svoboda},
    year={2023},
    eprint={2309.09007},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```
