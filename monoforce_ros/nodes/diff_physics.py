#!/usr/bin/env python

from time import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from monoforce.models.traj_predictor.dphys_config import DPhysConfig
from monoforce.models.traj_predictor.dphysics import DPhysics, generate_controls
from monoforce.ros import poses_to_marker, poses_to_path, gridmap_msg_to_numpy
from monoforce.transformations import pose_to_xyz_q
from nav_msgs.msg import Path
from grid_map_msgs.msg import GridMap
from ros_numpy import numpify
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray
from tf2_ros import TransformBroadcaster


class DiffPhysBase:
    def __init__(self,
                 gridmap_topic='/grid_map/terrain',
                 gridmap_layer='elevation',
                 robot_frame='base_link',
                 dphys_cfg: DPhysConfig = None,
                 max_age=0.5,
                 device='cpu',
                 dt=0.01):
        self.robot_frame = robot_frame
        self.dphys_cfg = dphys_cfg
        self.gridmap_layer = gridmap_layer
        self.gridmap_frame = None
        self.gridmap_center_frame = 'grid_map_link'
        self.dt = dt
        self.n_sim_steps = int(dphys_cfg.traj_sim_time / self.dt)
        self.max_age = max_age
        self.device = device
        self.n_sim_trajs = dphys_cfg.n_sim_trajs

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.path_cost_min = np.inf
        self.path_cost_max = -np.inf
        self.pose_step = int(0.5 / self.dt)  # publish poses with 0.5 [sec] step

        self.sampled_paths_pub = rospy.Publisher('/sampled_paths', MarkerArray, queue_size=1)
        self.lc_path_pub = rospy.Publisher('/lower_cost_path', Path, queue_size=1)
        self.path_costs_pub = rospy.Publisher('/path_costs', Float32MultiArray, queue_size=1)
        self.tf_broadcaster = TransformBroadcaster()

        # grid map subscriber to publish grid map tf
        self.gridmap_tf_sub = rospy.Subscriber(gridmap_topic, GridMap, self.gridmap_tf_callback)

    def gridmap_tf_callback(self, gridmap_msg):
        # publish grid map pose tf frame: create tf message from grid map pose
        grid_map_tf = TransformStamped()
        grid_map_tf.header.stamp = gridmap_msg.info.header.stamp
        grid_map_tf.header.frame_id = gridmap_msg.info.header.frame_id
        grid_map_tf.child_frame_id = self.gridmap_center_frame
        grid_map_tf.transform.translation.x = gridmap_msg.info.pose.position.x
        grid_map_tf.transform.translation.y = gridmap_msg.info.pose.position.y
        grid_map_tf.transform.translation.z = gridmap_msg.info.pose.position.z
        grid_map_tf.transform.rotation = gridmap_msg.info.pose.orientation
        self.tf_broadcaster.sendTransform(grid_map_tf)

    def init_poses(self):
        """
        Initialize robot poses.
        """
        xyz_q = np.zeros((self.n_sim_trajs, 7))
        xyz_q[:, 2] = 0.2  # robot's initial z coordinate
        xyz_q[:, 6] = 1.0  # quaternion w
        return xyz_q

    def init_controls(self):
        controls_front, _ = generate_controls(n_trajs=self.dphys_cfg.n_sim_trajs // 2,
                                              v_range=(self.dphys_cfg.vel_max / 2, self.dphys_cfg.vel_max),
                                              w_range=(-self.dphys_cfg.omega_max, self.dphys_cfg.omega_max),
                                              time_horizon=self.dphys_cfg.traj_sim_time, dt=self.dt)
        controls_back, _ = generate_controls(n_trajs=self.dphys_cfg.n_sim_trajs // 2,
                                             v_range=(-self.dphys_cfg.vel_max, -self.dphys_cfg.vel_max / 2),
                                             w_range=(-self.dphys_cfg.omega_max, self.dphys_cfg.omega_max),
                                             time_horizon=self.dphys_cfg.traj_sim_time, dt=self.dt)
        controls = torch.cat([controls_front, controls_back], dim=0)
        return controls

    def get_pose(self, target_frame, source_frame, stamp=None):
        if stamp is None:
            stamp = rospy.Time(0)
        try:
            tf = self.tf_buffer.lookup_transform(target_frame=target_frame, source_frame=source_frame,
                                                 time=stamp, timeout=rospy.Duration(1.0))
        except Exception as ex:
            rospy.logerr('Could not transform from %s to %s: %s.', source_frame, target_frame, ex)
            return None
        pose = np.array(numpify(tf.transform), dtype=np.float32).reshape((4, 4))
        return pose

    def publish_paths_and_costs(self, poses, path_costs, frame, stamp=None, pose_step=1):
        assert len(poses) == len(path_costs), 'xyz_q_np: %s, path_costs: %s' % (str(poses.shape), str(path_costs.shape))
        if stamp is None:
            stamp = rospy.Time.now()

        num_trajs = len(poses)

        # paths marker array
        marker_array = MarkerArray()
        red = np.array([1., 0., 0.])
        green = np.array([0., 1., 0.])
        path_costs_norm = (path_costs - self.path_cost_min) / (self.path_cost_max - self.path_cost_min)
        for i in range(num_trajs):
            # map path cost to color (lower cost -> greener, higher cost -> redder)
            color = green + (red - green) * path_costs_norm[i]
            marker_msg = poses_to_marker(poses[i][::pose_step], color=color)
            marker_msg.header.frame_id = frame
            marker_msg.header.stamp = stamp
            marker_msg.ns = 'paths'
            marker_msg.id = i
            marker_array.markers.append(marker_msg)

        # publish all sampled paths
        self.sampled_paths_pub.publish(marker_array)

        # publish lower cost path
        lower_cost_traj_i = np.argmin(path_costs)
        lower_cost_xyz_q = poses[lower_cost_traj_i]
        # rospy.logdebug('Min path cost: %.3f' % path_costs[lower_cost_traj_i])
        # rospy.logdebug('Max path cost: %.3f' % np.max(path_costs))
        rospy.loginfo('Publishing lower cost path of length: %d' % len(lower_cost_xyz_q[::pose_step]))
        path_msg = poses_to_path(lower_cost_xyz_q[::pose_step], stamp=stamp, frame_id=self.gridmap_frame)
        self.lc_path_pub.publish(path_msg)

        # publish path costs
        path_costs_msg = Float32MultiArray()
        path_costs_msg.data = path_costs
        self.path_costs_pub.publish(path_costs_msg)

    def gridmap_callback(self, gridmap_msg):
        assert isinstance(gridmap_msg, GridMap)
        # if message is stale do not process it
        dt = rospy.Time.now() - gridmap_msg.info.header.stamp
        if dt.to_sec() > self.max_age:
            rospy.logwarn(f'Stale grid map message received ({dt.to_sec():.1f} > {self.max_age} [sec]), skipping')
            return

        t0 = time()
        if self.gridmap_frame is None:
            self.gridmap_frame = gridmap_msg.info.header.frame_id

        # convert grid map to height map
        grid_map = gridmap_msg_to_numpy(gridmap_msg, self.gridmap_layer)
        assert not np.all(np.isnan(grid_map)) and np.all(np.isfinite(grid_map))
        assert grid_map.ndim == 2, 'Height map must be 2D'
        rospy.loginfo('Received height map of shape: %s' % str(grid_map.shape))

        grid_map_pose = numpify(gridmap_msg.info.pose).reshape((4, 4))
        robot_pose = self.get_pose(target_frame=self.gridmap_frame, source_frame=self.robot_frame,
                                   stamp=gridmap_msg.info.header.stamp)
        if robot_pose is None:
            rospy.logwarn('Could not get robot pose')
            return
        robot_pose_wrt_gridmap = np.linalg.inv(grid_map_pose) @ robot_pose
        robot_xyz_q_wrt_gridmap = pose_to_xyz_q(robot_pose_wrt_gridmap)
        t1 = time()
        rospy.logdebug('Grid map preprocessing took %.3f [sec]' % (t1 - t0))

        # predict path
        grid_maps = [grid_map for _ in range(self.n_sim_trajs)]
        xyz_qs_init = np.repeat(robot_xyz_q_wrt_gridmap[None, :], self.n_sim_trajs, axis=0)
        with torch.no_grad():
            t2 = time()
            xyz_qs, path_costs = self.predict_paths(grid_maps, xyz_qs_init)
            t3 = time()
            rospy.loginfo('Path prediction took %.3f [sec]' % (t3 - t2))

        # update path cost bounds
        if path_costs is not None:
            self.path_cost_min = min(self.path_cost_min, torch.min(path_costs).item())
            self.path_cost_max = max(self.path_cost_max, torch.max(path_costs).item())
            rospy.logdebug('Path cost min: %.3f' % self.path_cost_min)
            rospy.logdebug('Path cost max: %.3f' % self.path_cost_max)

        # publish paths
        if xyz_qs is not None:
            t4 = time()
            xyz_qs_np = xyz_qs.cpu().numpy()
            path_costs_np = path_costs.cpu().numpy()
            self.publish_paths_and_costs(xyz_qs_np, path_costs_np, stamp=gridmap_msg.info.header.stamp,
                                         frame=self.gridmap_center_frame, pose_step=self.pose_step)
            rospy.logdebug('Paths publishing took %.3f [sec]' % (time() - t4))

    @staticmethod
    def spin():
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            pass

    def predict_paths(self, grid_maps, xyz_qs_init):
        raise NotImplementedError

class DPhysEngine(DiffPhysBase):
    def __init__(self,
                 dphys_cfg: DPhysConfig,
                 gridmap_topic='/grid_map/terrain',
                 gridmap_layer='elevation',
                 robot_frame='base_link',
                 max_age=0.5,
                 dt=0.01,
                 device='cpu'):
        super().__init__(dphys_cfg=dphys_cfg, gridmap_topic=gridmap_topic, gridmap_layer=gridmap_layer, robot_frame=robot_frame,
                         max_age=max_age, device=device, dt=dt)
        self.dphysics = DPhysics(dphys_cfg, device=device)
        self.track_vels = self.init_controls()
        # grid map subscriber
        self.gridmap_sub = rospy.Subscriber(gridmap_topic, GridMap, self.gridmap_callback)

    def predict_paths(self, grid_maps, xyz_qs_init):
        assert len(grid_maps) == len(xyz_qs_init) == self.n_sim_trajs
        # for grid_map in grid_maps:
        #     assert isinstance(grid_map, np.ndarray)
        #     assert grid_map.shape[0] == grid_map.shape[1]
        grid_maps = torch.as_tensor(grid_maps, dtype=torch.float32, device=self.device)
        assert grid_maps.shape[0] == self.n_sim_trajs
        controls = torch.as_tensor(self.track_vels, dtype=torch.float32, device=self.device)
        assert controls.shape == (self.n_sim_trajs, self.n_sim_steps, 2), \
            f'controls shape: {controls.shape} != {(self.n_sim_trajs, self.n_sim_steps, 2)}'

        # initial state
        x = torch.as_tensor(xyz_qs_init[:, :3], dtype=torch.float32, device=self.device)
        xd = torch.zeros_like(x)
        R = torch.as_tensor(Rotation.from_quat(xyz_qs_init[:, 3:]).as_matrix(), dtype=torch.float32, device=self.device)
        R.repeat(x.shape[0], 1, 1)
        omega = torch.zeros_like(x)
        state0 = (x, xd, R, omega)

        # simulate trajectories
        states, forces = self.dphysics(grid_maps, controls=controls, state=state0)
        Xs, Xds, Rs, Omegas = states
        assert Xs.shape == (self.n_sim_trajs, self.n_sim_steps, 3)
        assert Rs.shape == (self.n_sim_trajs, self.n_sim_steps, 3, 3)

        # convert rotation matrices to quaternions
        poses = torch.zeros((self.n_sim_trajs, self.n_sim_steps, 4, 4), device=self.device)
        poses[:, :, :3, 3] = Xs
        poses[:, :, :3, :3] = Rs
        poses[:, :, 3, 3] = 1.0
        assert not torch.any(torch.isnan(poses))

        # compute path costs: interaction forces-based
        # F_normals, F_frictions = forces
        # assert F_normals.shape == (self.n_sim_trajs, self.n_sim_steps, len(self.dphys_cfg.robot_points), 3)
        # assert F_frictions.shape == (self.n_sim_trajs, self.n_sim_steps, len(self.dphys_cfg.robot_points), 3)
        # path_costs = F_normals.norm(dim=-1).std(dim=-1).std(dim=-1)
        # assert not torch.any(torch.isnan(path_costs))
        # assert poses.shape == (self.n_sim_trajs, self.n_sim_steps, 4, 4)
        # assert path_costs.shape == (self.n_sim_trajs,)

        # compute path costs: inclination-based
        Rs_pred = states[2].cpu().reshape(-1, 3, 3)  # (n_trajs * time_horizon, 3, 3)
        rpy = torch.as_tensor(Rotation.from_matrix(Rs_pred).as_euler('xyz'))  # (n_trajs * time_horizon, 3)
        roll, pitch = rpy[:, 0].reshape(self.n_sim_trajs, -1), rpy[:, 1].reshape(self.n_sim_trajs, -1)  # (n_trajs, time_horizon)
        path_costs = roll.abs().mean(dim=-1) + pitch.abs().mean(dim=-1)  # (n_trajs,)

        return poses, path_costs


def main():
    rospy.init_node('diff_physics', anonymous=True, log_level=rospy.DEBUG)

    robot = rospy.get_param('~robot', 'tradr')
    dphys_cfg = DPhysConfig(robot=robot)
    robot_frame = rospy.get_param('~robot_frame', 'base_link')
    gridmap_topic = rospy.get_param('~gridmap_topic')
    gridmap_layer = rospy.get_param('~gridmap_layer', 'elevation')
    max_age = rospy.get_param('~max_age', 0.5)
    device = rospy.get_param('~device', 'cuda' if torch.cuda.is_available() else 'cpu')

    node = DPhysEngine(dphys_cfg=dphys_cfg,
                       robot_frame=robot_frame,
                       gridmap_topic=gridmap_topic,
                       gridmap_layer=gridmap_layer,
                       max_age=max_age,
                       device=device)
    node.spin()


if __name__ == '__main__':
    main()
