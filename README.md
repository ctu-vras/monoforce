# MonoForce: Learnable Image-conditioned Physics Engine

[![Arxiv](http://img.shields.io/badge/paper-arxiv-critical.svg?style=plastic)](https://arxiv.org/abs/2502.10156)
[![IROS-2024](http://img.shields.io/badge/paper-IROS_2024-blue.svg?style=plastic)](https://ieeexplore.ieee.org/abstract/document/10801353)
[![Arxiv](http://img.shields.io/badge/paper-arxiv_IROS-critical.svg?style=plastic)](https://arxiv.org/abs/2309.09007)
[![ICML-2024-Diff-XYZ](http://img.shields.io/badge/paper-ICML_2024_Diff_XYZ-green.svg?style=plastic)](https://differentiable.xyz/papers-2024/paper_30.pdf)

[![Video](http://img.shields.io/badge/video-13mins-blue.svg?style=plastic)](https://drive.google.com/file/d/1XP0o5xvUlE1_pGOX9tKGPCKOJVdANM_H/view?usp=sharing)
[![Video](http://img.shields.io/badge/video-5mins-blue.svg?style=plastic)](https://drive.google.com/file/d/1fJKmqy3yXl6jpsRioHqYbRtveDYc4cco/view?usp=drive_link)
[![Video](http://img.shields.io/badge/video-1min-blue.svg?style=plastic)](https://drive.google.com/file/d/1tTt1Oi5k1jKPDYn3CnzArhV3NPSNxKvD/view?usp=sharing)
[![Poster](http://img.shields.io/badge/poster-A0-blue.svg?style=plastic)](https://docs.google.com/presentation/d/1A9yT6MC-B9DdzMdzCZ44Y8nVHtBHWxpLAUOZKS_GUcU/edit?usp=sharing)
[![Data](http://img.shields.io/badge/data-ROUGH-blue.svg?style=plastic)](https://drive.google.com/drive/folders/1nli-4YExqcBhl0mPNRUjSiNecX4yIcme?usp=sharing)

<img src="./monoforce/docs/imgs/qualitative_results_v2.jpg" width="800"/>

Robot-terrain interaction prediction from RGB camera images as input:
- predicted trajectory,
- terrain shape and properties,
- interaction forces and contacts.

<img src="./monoforce/docs/imgs/examples/ramp_success.png" width="200"/> <img src="./monoforce/docs/imgs/examples/high_grass2.png" width="200"/> <img src="./monoforce/docs/imgs/examples/wall3.png" width="200"/> <img src="./monoforce/docs/imgs/examples/snow.png" width="200"/>

Examples of predicted trajectories and autonomous traversal through vegetation:

<p>
  <a href="https://www.youtube.com/watch?v=JGi-OzTBG1k">
    <img src="./monoforce/docs/imgs/demo_oru.png" alt="video link" width="300">
  </a>
  <a href="https://drive.google.com/file/d/1TTNTyqZnObtdE_PdCc2GprszphnE3hxS/view?usp=drive_link">
    <img src="./monoforce/docs/imgs/park_navigation_video_teaser.jpg" alt="video link" width="500">
  </a>
</p>

## Table of Contents
- [Installation Instructions](./monoforce/docs/INSTALL.md)
- [Data](./monoforce/docs/DATA.md)
- [Terrain Encoder](./monoforce/docs/TERRAIN_ENCODER.md)
- [Physics Engine](./monoforce/docs/PHYSICS_ENGINE.md)
- [Overview](#overview)
- [ROS Integration](#ros2-integration)
- [Training](#training)
- [Navigation](#navigation)
- [Citation](#citation)

## Overview

<img src="./monoforce/docs/imgs/pipeline.png" width="800"/>

The MonoForce pipeline consists of the Terrain Encoder and the Physics Engine.
Given input RGB images and cameras calibration the Terrain Encoder predicts terrain properties.
Then the differentiable Physics Engine simulates robot trajectory and interaction forces on the predicted terrain
for a provided control sequence.

## ROS2 Integration

<img src="./monoforce/docs/imgs/monoforce.gif" width="800"/>

We provide ROS2 nodes for both the trained Terrain Encoder and the Physics Engine.
Depending on the application, they can be run separately.
- **Terrain Encoder**: The Terrain Encoder node takes RGB images and camera calibration as input and predicts terrain properties.
```commandline
ros2 launch monoforce terrain_encoder.launch.py
```
- **Physics Engine**: The Physics Engine node takes the predicted terrain properties and as input
(the control commands are predefined) and predicts the robot trajectories and interaction forces.
```commandline
ros2 launch monoforce physice_engine.launch.py
```
- They are also integrated into a single **MonoForce** node that can be launched with the following command:
```commandline
ros2 launch monoforce monoforce.launch.py
```

## Training

The following terrain properties are predicted by the model:
- **Elevation**: the terrain shape.
- **Friction**: the friction coefficient between the robot and the terrain.
- **Stiffness**: the terrain stiffness.
- **Damping**: the terrain damping.

<img src="./monoforce/docs/imgs/training.jpg" width="800"/>

An example of the predicted elevation and friction maps (projected to camera images):
<p>
  <a href="https://drive.google.com/file/d/15Uo82hwE_OiRHsuGd0-9qcvrYOXsosn0/view?usp=drive_link">
  <img src="./monoforce/docs/imgs/friction_prediction_tradr.png" alt="video link" width="800">
  </a>
</p>
One can see that the model predicts the friction map with
higher values for road areas and with the smaller value
for grass where the robot could have less traction.


## Navigation

Navigation method with MonoForce predicting terrain properties
and possible robot trajectories from RGB images and control inputs.
The package is used as robot-terrain interaction and path planning pipeline.

<p>
  <a href="https://drive.google.com/file/d/1mqKEh_3VHZo4kDcJXP572SD1BVw37hSf/view?usp=drive_link">
  <img src="monoforce/docs/imgs/forest_navigation_video_teaser.png" alt="video link" width="800">
  </a>
</p>

Navigation consists of the following stages:
- **Terrain prediction**: The Terrain Encoder is used to estimate terrain properties.
- **Trajectories simulation**: The Physics Engine is used to shoot the robot trajectories.
- **Trajectory selection**: The trajectory with the smallest cost based on robot-terrain interaction forces
and waypoint distance is selected.
- **Control**: The robot is controlled to follow the selected trajectory.

## Citation

Consider citing the papers if you find the work relevant to your research:

```bibtex
@inproceedings{agishev2024monoforce,
    title={MonoForce: Self-supervised Learning of Physics-informed Model for Predicting Robot-terrain Interaction},
    author={Ruslan Agishev and Karel Zimmermann and Vladimír Kubelka and Martin Pecka and Tomáš Svoboda},
    booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems - IROS},
    year={2024},
    eprint={2309.09007},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2309.09007},
    doi={10.1109/IROS58592.2024.10801353},
}
```

```bibtex
@inproceedings{agishev2024endtoend,
    title={End-to-end Differentiable Model of Robot-terrain Interactions},
    author={Ruslan Agishev and Vladim{\'\i}r Kubelka and Martin Pecka and Tomas Svoboda and Karel Zimmermann},
    booktitle={ICML 2024 Workshop on Differentiable Almost Everything: Differentiable Relaxations, Algorithms, Operators, and Simulators},
    year={2024},
    url={https://openreview.net/forum?id=XuVysF8Aon}
}
```
