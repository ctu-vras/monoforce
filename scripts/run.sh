#!/bin/bash

python run.py --img-paths ../config/data_sample/marv/images/1727353472_601360559_camera_left.png \
                          ../config/data_sample/marv/images/1727353472_601360559_camera_front.png \
                          ../config/data_sample/marv/images/1727353472_601360559_camera_right.png \
                          ../config/data_sample/marv/images/1727353472_601360559_camera_rear.png \
              --cameras camera_left camera_front camera_right camera_rear \
              --calibration-path ../config/data_sample/marv/calibration/ \
              --lss_cfg_path ../config/lss_cfg.yaml \
              --model_path ../config/weights/lss/val.pth
