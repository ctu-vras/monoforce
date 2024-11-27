#!/usr/bin/env python

import os
import numpy as np
import rospy
from grid_map_msgs.msg import GridMap
from monoforce.dphys_config import DPhysConfig
from monoforce.ros import height_map_to_gridmap_msg
from monoforce.cloudproc import estimate_heightmap, filter_grid, filter_range
from monoforce.utils import position
from ros_numpy import numpify
from sensor_msgs.msg import PointCloud2
import rospkg
import tf2_ros
from time import time
from message_filters import ApproximateTimeSynchronizer, Subscriber

lib_path = rospkg.RosPack().get_path('monoforce').replace('monoforce_ros', 'monoforce')


class HeightMapEstimator:
    def __init__(self, cfg: DPhysConfig,
                 pts_topics=['points'],
                 robot_frame='base_link',
                 ground_frame='base_footprint'):
        self.cfg = cfg
        self.robot_frame = robot_frame
        self.ground_frame = ground_frame
        self.robot_clearance= None

        # tf buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # grid map publisher
        self.gridmap_pub = rospy.Publisher('/grid_map/terrain', GridMap, queue_size=1)

        # max message delay
        self.max_age = rospy.get_param('~max_age', 1.0)

        # point cloud subscribers
        self.pts_subs = []
        for topic in pts_topics:
            self.pts_subs.append(Subscriber(topic, PointCloud2))
        self.pts_sync = ApproximateTimeSynchronizer(self.pts_subs, queue_size=1, slop=1.0)
        self.pts_sync.registerCallback(self.pts_callback)

    def get_robot_clearance(self):
        # base_link -> base_footprint
        try:
            tf = self.tf_buffer.lookup_transform(target_frame=self.ground_frame,
                                                 source_frame=self.robot_frame,
                                                 time=rospy.Time(0),
                                                 timeout=rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.logwarn('Could not get transform from %s to %s due to %s' % (self.robot_frame, self.ground_frame, ex))
            return None
        Tr = numpify(tf.transform).reshape((4, 4))
        robot_clearance = Tr[2, 3]
        assert robot_clearance >= 0, 'Robot clearance must be non-negative (got %.3f)' % robot_clearance
        return robot_clearance

    def pts_callback(self, *msgs):
        msg = msgs[0]
        # if message is stale do not process it
        dt = rospy.Time.now() - msg.header.stamp
        if dt.to_sec() > self.max_age:
            rospy.logdebug(
                f'Stale image messages received ({dt.to_sec():.3f} > {self.max_age} [sec]), skipping')
            return

        t0 = time()
        if self.robot_clearance is None:
            self.robot_clearance = self.get_robot_clearance()
            if self.robot_clearance is None:
                rospy.logwarn('Could not get robot clearance')
                return
        else:
            rospy.logdebug('Robot clearance: %.3f' % self.robot_clearance)

        # convert messages to points
        points = self.msgs_to_points(msgs)
        if points is None:
            rospy.logwarn('Could not convert messages to points')
            return
        rospy.logdebug('Points shape: %s' % str(points.shape))

        # estimate height map
        hm = estimate_heightmap(points, r_min=self.cfg.r_min, d_max=self.cfg.d_max,
                                grid_res=self.cfg.grid_res,
                                h_max=self.cfg.h_max)
        if hm is None:
            rospy.logwarn('Could not estimate height map')
            return
        height = hm[0]
        rospy.logdebug('Estimated height map shape: %s' % str(height.shape))
        rospy.logdebug('HM estimation time: %.3f' % (time() - t0))

        # publish grid map
        t1 = time()
        grid_msg = height_map_to_gridmap_msg(height - self.robot_clearance, self.cfg.grid_res,
                                             xyz=np.array([0., 0., 0.]), q=np.array([0., 0., 0., 1.]))
        grid_msg.info.header.stamp = msg.header.stamp
        grid_msg.info.header.frame_id = self.robot_frame
        self.gridmap_pub.publish(grid_msg)
        rospy.logdebug('Height map publishing took %.3f' % (time() - t1))

    def msgs_to_points(self, msgs):
        all_points = []
        for msg in msgs:
            cloud = numpify(msg)
            if cloud.ndim > 1:
                cloud = cloud.reshape(-1)
            points = position(cloud)

            # apply range filter
            points = filter_range(points, min=self.cfg.r_min, max=np.sqrt(2) * self.cfg.d_max)

            # apply grid filter
            points = filter_grid(points, self.cfg.grid_res)

            # transform points to robot frame
            try:
                tf = self.tf_buffer.lookup_transform(target_frame=self.robot_frame,
                                                     source_frame=msg.header.frame_id,
                                                     time=msg.header.stamp,
                                                     timeout=rospy.Duration(1.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
                rospy.logwarn('Could not get transform from %s to %s due to %s' % (msg.header.frame_id, self.robot_frame, ex))
                return None

            Tr = numpify(tf.transform).reshape((4, 4))
            points = np.matmul(Tr[:3, :3], points.T).T + Tr[:3, 3]
            all_points.append(points)

        all_points = np.concatenate(all_points, axis=0)
        rospy.logdebug('Points shape: %s' % str(all_points.shape))
        return all_points


def main():
    rospy.init_node('hm_estimator', anonymous=True, log_level=rospy.INFO)

    cfg = DPhysConfig()
    cfg.hm_interp_method = rospy.get_param('~hm_interp_method', 'nearest')
    # cfg.to_rosparam()

    pts_topics = rospy.get_param('~pts_topics')
    robot_frame = rospy.get_param('~robot_frame', 'base_link')
    ground_frame = rospy.get_param('~ground_frame', 'base_footprint')
    node = HeightMapEstimator(cfg=cfg, pts_topics=pts_topics, robot_frame=robot_frame, ground_frame=ground_frame)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down')


if __name__ == '__main__':
    main()
