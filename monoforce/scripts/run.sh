#!/bin/bash

python run --img-paths ../config/data_sample/husky/images/1666877643_267394304_camera_front.png \
                       ../config/data_sample/husky/images/1666877643_267394304_camera_left.png \
                       ../config/data_sample/husky/images/1666877643_267394304_camera_right.png \
                       ../config/data_sample/husky/images/1666877643_267394304_camera_rear.png \
           --cameras camera_front camera_left camera_right camera_rear \
           --calibration-path ../config/data_sample/husky/calibration/ \
           --lss_cfg_path ../config/lss_cfg.yaml \
           --model_path ../config/weights/lss/lss.pt \
           --linear-vel 1.0 --angular-vel 1.0
