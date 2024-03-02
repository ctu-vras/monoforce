## Dataset
The data is available at 
[http://subtdata.felk.cvut.cz/robingas/data/](http://subtdata.felk.cvut.cz/robingas/data/).

- Semi-supervised traversability data generated using lidar odometry and IMU.
- The dataset additionally contains:
  - camera images,
  - calibration data,
  - and RGB colors projected from cameras onto the point clouds.

The traversability dataset has the following structure:
```commandline
<experiment-date-place>
└── <robot name>
    └── <sequence name>_trav/
        ├── calibration
        │   ├── cameras
        │   ├── img_statistics.yaml
        |   └── transformations.yaml
        ├── cloud_colors
        ├── clouds
        ├── images
        ├── trajectories
        └── traj_poses.csv
```
Folders names example:
[22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/](http://subtdata.felk.cvut.cz/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/).
One can download a sequence folder by running:
```commandline
scp -r <username>@subtdata.felk.cvut.cz:/data/robingas/data/<experiment-date-place>/<robot-name>/<sequence-name>_trav/ .
```

![](./imgs/lidar_imu_trav.png)

The semi-supervised data is stored in the `clouds` folder.
The point clouds traversed by a robot are labeled with the recorded onboard IMU measurements.
Please have a look at the
[video](https://drive.google.com/file/d/1CmLwgTUFmKrMXm5hG5n1Bz0XBZqLNifc/view?usp=drive_link)
for the data preview from a sequence recorded with a tracked robot in a forest environment.

In order to generate the traversability data from a prerecorded bag file, please run
(*note, that the topic names could be different depending on a bag file):
```commandline
cd ./scripts/data/
./save_clouds_and_trajectories_from_bag --bag-paths /path/to/dataset/<experiment-date-place>/<robot-name>/<sequence-name>.bag \
                                                    /path/to/dataset/<experiment-date-place>/<robot-name>/<sequence-name>_loc.bag \
                                                    --cloud-topics /os_cloud_node/destaggered_points \
                                                    --imu-topics /imu/data \
                                                    --robot-model 'Box()' --discard-model 'Box()' \
                                                    --input-step 50 --visualize False --save-data True
```

To obtain camera-lidar calibration data from a bag file and to project RGB colors from cameras onto the point clouds,
please run:
```commandline
cd ./scripts/data/
./add_calibrations_colors --bag-path /path/to/dataset/<experiment-date-place>/<robot-name>/<sequence-name>.bag \
                                     --lidar-topic /os_cloud_node/destaggered_points \
                                     --camera-topics /camera_fisheye_rear/image_color/compressed \
                                                     /camera_fisheye_front/image_color/compressed \
                                                     /camera_right/image_color/compressed \
                                                     /camera_left/image_color/compressed \
                                                     /camera_up/image_color/compressed \
                                     --camera-info-topics /camera_fisheye_front/camera_info \
                                                         /camera_fisheye_rear/camera_info \
                                                         /camera_right/camera_info \
                                                         /camera_left/camera_info \
                                                         /camera_up/camera_info \
                                     --visualize False --save-data True
```

Camera views examples:

Colored point cloud            |            Front-facing camera            |        Up-facing camera        
:-----------------------------:|:-----------------------------------------:|:------------------------------:
![](./imgs/rgb_cloud.png) | ![](./imgs/camera_fisheye_front.png) | ![](./imgs/camera_up.png)

To explore the data, please run:
```commandline
python -m monoforce.datasets.data
```