## Terrain Encoder

The Terrain Encoder is a model that predicts robot's supporting terrain from input RGB images,
[video](https://drive.google.com/file/d/17GtA_uLyQ2o3tHiBuhxenZ0En7SzLAad/view?usp=sharing).

<img src="imgs/hm_prediction_demo.png" height="280"/> <img src="imgs/images_to_heightmap.png" height="280"/>

### Training

![](./imgs/architecture_v3.png)

- The Terrain Encoder takes as input calibrated RGB-camera images and predicts terrain properties around the robot
in the bird-eye-view format. We utilize the [Lift-Splat-Shoot (LSS)](https://github.com/nv-tlabs/lift-splat-shoot) architecture as the Terrain Encoder.
- The Differentiable Physics Engine uses the estimated terrain properties to predict
the robot's trajectory.
- The predicted trajectory is compared to a ground-truth one (provided from SLAM, odometry, or GPS).
- Lidar scans are used in order to provide initial height map estimates that are refined during training.
- Semantic labels are used to infer the information that the objects known to be rigid (not-deformable)
should remain in the heightmap.
