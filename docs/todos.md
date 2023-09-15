## Task list

- [x] Run terrain learning with KKT data
- [x] Run terrain learning with RobinGas data
- [x] Fix heightmap construction from RobinGas data
- [x] Fix sampled trajectory loss
- [x] Add saving images to the RobinGas data
- [x] Images color projection to point clouds
- [x] Add saving velocities to the RobinGas data
- [x] Add possibility to have time dependent tracks velocities
- [x] RobinGas data documentation
- [ ] Differentiable physics documentation
- [ ] Terrain predictor training pipeline documentation
- [x] Height map estimates generated from point clouds correction with trajectories L2 loss
- [x] Save corrected height maps (masked with the traversed area)
- [x] Define terrain predictor input data format (RGB images with estimated height maps from point clouds as reularization)
- [x] Train terrain predictor with corrected (optimized) height maps
- [ ] Script for creating KKT format data
- [x] Define terrain predictor evaluation metrics (L2-loss between predicted trajectory, given control commands and predicted height map and ground truth trajectory).
- [x] Evaluate the terrain predictor with the KKT and RobinGas data, compare to the KKT traversability model.
