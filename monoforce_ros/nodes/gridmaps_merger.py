#!/usr/bin/env python

import os
import numpy as np
import rospy
from grid_map_msgs.msg import GridMap
from monoforce.cloudproc import merge_heightmaps
from monoforce.config import DPhysConfig
from monoforce.ros import height_map_to_gridmap_msg, gridmap_msg_to_numpy
from monoforce.transformations import transform_cloud
from ros_numpy import numpify
import rospkg
import tf2_ros

lib_path = rospkg.RosPack().get_path('monoforce').replace('monoforce_ros', 'monoforce')


class GridMapsMerger:
    def __init__(self, cfg: DPhysConfig,
                 gridmap_topic='grid_map/terrain',
                 robot_frame='base_link',
                 map_frame='map'):
        self.cfg = cfg
        self.robot_frame = robot_frame
        self.map_frame = map_frame

        # tf buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # grid map publisher
        self.gridmap_pub = rospy.Publisher('/grid_map_merged/terrain', GridMap, queue_size=1)

        # max message delay
        self.max_age = rospy.get_param('~max_age', 1.0)
        
        self.merged_points = None

        # heightmap subscriber
        self.gridmap_sub = rospy.Subscriber(gridmap_topic, GridMap, self.gridmap_callback, queue_size=1)

    def get_map_pose(self, frame, stamp=None):
        if stamp is None:
            stamp = rospy.Time(0)
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, frame, stamp, rospy.Duration(1.0))
        except Exception as ex:
            rospy.logerr('Could not transform from %s to %s: %s.', frame, self.map_frame, ex)
            return None
        pose = np.array(numpify(tf.transform), dtype=np.float32).reshape((4, 4))
        return pose

    def gridmap_callback(self, msg):
        assert isinstance(msg, GridMap)
        # if message is stale do not process it
        dt = rospy.Time.now() - msg.info.header.stamp
        if dt.to_sec() > self.max_age:
            rospy.logwarn(f'Stale heightmap messages received ({dt.to_sec():.3f} > {self.max_age} [sec]), skipping')
            return

        height = gridmap_msg_to_numpy(msg)
        rospy.logdebug('Heightmap shape: %s', height.shape)

        # transform cloud to map frame
        pose = self.get_map_pose(msg.info.header.frame_id)
        W, H = height.shape
        x, y = np.meshgrid(np.linspace(-msg.info.length_x / 2, msg.info.length_x / 2, W),
                           np.linspace(-msg.info.length_y / 2, msg.info.length_y / 2, H))
        points = np.stack([x.flatten(), y.flatten(), height.flatten()], axis=1)
        assert points.shape == (W * H, 3)
        points = transform_cloud(points, pose)

        # merge heightmaps
        if self.merged_points is None:
            self.merged_points = points.copy()
        else:
            self.merged_points = merge_heightmaps(points.copy(), self.merged_points.copy(), self.cfg.grid_res)

        H, W = int(2 * self.cfg.d_max / self.cfg.grid_res), int(2 * self.cfg.d_max / self.cfg.grid_res)
        height_merged = self.merged_points[:, 2].reshape(H, W).T
        rospy.logdebug('Merged heightmap shape: %s', height_merged.shape)

        # publish heightmap as grid map
        stamp = rospy.Time.now()
        grid_msg = height_map_to_gridmap_msg(height_merged, self.cfg.grid_res)
        grid_msg.info.header.stamp = stamp
        grid_msg.info.header.frame_id = self.robot_frame
        self.gridmap_pub.publish(grid_msg)

    def spin(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print('Shutting down')
    

def main():
    rospy.init_node('gridmaps_merger', anonymous=True, log_level=rospy.DEBUG)

    cfg = DPhysConfig()
    config_path = rospy.get_param('~config_path', os.path.join(lib_path, 'config/dphys_cfg.yaml'))
    assert os.path.isfile(config_path), 'Config file %s does not exist' % config_path
    cfg.from_yaml(config_path)

    gridmap_topic = rospy.get_param('~gridmap_topic', '/grid_map/terrain')
    robot_frame = rospy.get_param('~robot_frame', 'base_link')
    map_frame = rospy.get_param('~map_frame', 'map')
    node = GridMapsMerger(cfg=cfg, gridmap_topic=gridmap_topic, robot_frame=robot_frame, map_frame=map_frame)
    node.spin()


if __name__ == '__main__':
    main()
