#!/bin/bash

python run --img-paths ../config/data_sample/tradr/images/1666267171_394104004_camera_front.png \
                       ../config/data_sample/tradr/images/1666267171_394104004_camera_left.png \
                       ../config/data_sample/tradr/images/1666267171_394104004_camera_right.png \
                       ../config/data_sample/tradr/images/1666267171_394104004_camera_rear_left.png \
                       ../config/data_sample/tradr/images/1666267171_394104004_camera_rear_right.png \
           --cameras camera_front camera_left camera_right camera_rear_left camera_rear_right \
           --calibration-path ../config/data_sample/tradr/calibration/ \
           --lss_cfg_path ../config/lss_cfg_tradr.yaml --model_path ../config/weights/lss/lss_robingas_tradr.pt --dphys_cfg_path ../config/dphys_cfg.yaml \
           --linear-vel 1.0 --angular-vel -0.1
