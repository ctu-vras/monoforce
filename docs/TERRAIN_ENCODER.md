## Terrain Encoder

The Terrain Encoder is a model that predicts robot's supporting terrain from input RGB images.
The demo video is available via the [link](https://drive.google.com/file/d/17GtA_uLyQ2o3tHiBuhxenZ0En7SzLAad/view?usp=sharing).

<img src="imgs/hm_prediction_demo.png" height="280"/> <img src="imgs/images_to_heightmap.png" height="280"/>

### Training

![](./imgs/terrain_encoder_training.png)

1. Using the Differentiable Physics module, the terrain shape under the robot trajectory is optimized in order to match the ground-truth trajectory as closely as possible.
2. The optimized terrain shape is used as a label to train the terrain shape predictor. This model takes as input an RGB-image and predicts the shape of the supporting terrain in front of a robot.
We utilize the [Lift-Splat-Shoot (LSS)](https://github.com/nv-tlabs/lift-splat-shoot) model as the Terrain Encoder.
3. Lidar scans are used in order to provide initial height map estimates during training.

To train the LSS model, please run:
```commandline
cd scripts/
python train
```

### Weights

The pretrained weights for the LSS terrain encoder can be downloaded from:
- RobinGas: [lss_robingas_2024_03_04_09_42_47/train_lss.pt](https://drive.google.com/file/d/168W8ftzlLFOquIb1mLTrSkjgMLHDOks0/view?usp=sharing)
- RELLIS-3D: [lss_rellis3d_2024_03_06_16_07_52/train_lss.pt](https://drive.google.com/file/d/12WUNFXFHsm3hM1Ov-Ap1yRybOif6-Vi4/view?usp=sharing)