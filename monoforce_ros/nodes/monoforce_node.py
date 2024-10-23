#!/usr/bin/env python

import os
import torch
import numpy as np
import rospy
from grid_map_msgs.msg import GridMap
from monoforce.ros import height_map_to_gridmap_msg
from monoforce.utils import read_yaml, timing
from terrain_encoder import TerrainEncoder
import rospkg


class MonoForce(TerrainEncoder):
    def __init__(self, lss_cfg):
        super(MonoForce, self).__init__(lss_cfg)


def main():
    rospy.init_node('monoforce', anonymous=True, log_level=rospy.DEBUG)
    lib_path = rospkg.RosPack().get_path('monoforce').replace('monoforce_ros', 'monoforce')
    rospy.loginfo('MonoForce library path: %s' % lib_path)

    # load configs
    lss_config_path = rospy.get_param('~lss_config_path', os.path.join(lib_path, 'config/lss_cfg.yaml'))
    assert os.path.isfile(lss_config_path), 'LSS config file %s does not exist' % lss_config_path
    lss_cfg = read_yaml(lss_config_path)

    node = MonoForce(lss_cfg=lss_cfg)
    node.start()
    node.spin()


if __name__ == '__main__':
    main()
