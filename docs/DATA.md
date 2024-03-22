## Dataset

Self-supervised traversability data generated using lidar SLAM.
The data sequences are available at 
[https://drive.google.com/drive/folders/1TdEUQ5m5la3Q8DCrRzxeDJKlrYyCMkb9?usp=sharing](https://drive.google.com/drive/folders/1TdEUQ5m5la3Q8DCrRzxeDJKlrYyCMkb9?usp=sharing).

The dataset contains:
  - point clouds, 
  - camera images,
  - calibration data: camera-lidar extrinsics, camera intrinsics, and distortion parameters,
  - localization data: robot poses for each point cloud stamp,
  - robot's future trajectories data (10 seconds long).

The traversability dataset has the following structure:
```commandline
<sequence name>
    ├── calibration
    │   ├── cameras
    │   ├── img_statistics.yaml
    |   └── transformations.yaml
    ├── clouds
    ├── images
    ├── poses
    ├── terrain
    │   ├── lidar
    │   └── traj
    ├── trajectories
    └── visuals
```

The point clouds (located in the `clouds` folder) are segmented by the robot's footprint trajectory.
Please have a look at the
[video](https://drive.google.com/file/d/1CmLwgTUFmKrMXm5hG5n1Bz0XBZqLNifc/view?usp=drive_link)
for the data preview from a sequence recorded with the tracked robot in a forest environment.

Camera views examples:

Colored point cloud            |            Front-facing camera            |        Up-facing camera        
:-----------------------------:|:-----------------------------------------:|:------------------------------:
![](./imgs/rgb_cloud.png) | ![](./imgs/camera_fisheye_front.png) | ![](./imgs/camera_up.png)

To explore the data, please run:
```commandline
python -m monoforce.datasets.robingas
```

### Data Sample

The [LSS](https://github.com/nv-tlabs/lift-splat-shoot) model training data example include:
- input RGB images,
- terrain heightmap estimated from lidar,
- robot footprint trajectory,
- point cloud generated from camera frustums.

![](./imgs/lss_data.jpg)
