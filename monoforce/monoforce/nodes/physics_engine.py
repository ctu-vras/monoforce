#!/usr/bin/env python

from collections import deque
from time import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from monoforce.models.physics_engine.engine.engine import DPhysicsEngine, PhysicsState
from monoforce.configs import WorldConfig, RobotModelConfig, PhysicsEngineConfig
from monoforce.models.physics_engine.utils.environment import make_x_y_grids
from monoforce.models.physics_engine.utils.torch_utils import set_device
from monoforce.models.physics_engine.engine.engine_state import vectorize_iter_of_states as vectorize_states
from monoforce.ros import poses_to_marker, gridmap_msg_to_numpy, pose_to_matrix
from monoforce.transformations import pose_to_xyz_q

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.impl.logging_severity import LoggingSeverity

import tf2_ros
from geometry_msgs.msg import TransformStamped
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray
from tf2_ros import TransformBroadcaster


class DiffPhysEngineNode(Node):
    def __init__(self):
        super().__init__('physics_engine')
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('num_robots', 32)
        self.declare_parameter('traj_sim_time', 5.0)
        self.declare_parameter('grid_res', 0.1)
        self.declare_parameter('max_coord', 6.4)
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('gridmap_topic', '/terrain/grid_map')
        self.declare_parameter('gridmap_layer', 'elevation')
        self.declare_parameter('max_age', 0.5)

        self.device = set_device(self.get_parameter('device').value)
        self._logger.set_level(LoggingSeverity.DEBUG)

        self.robot_config = RobotModelConfig()
        max_coord = self.get_parameter('max_coord').value
        grid_res = self.get_parameter('grid_res').value
        num_robots = self.get_parameter('num_robots').value
        x_grid, y_grid = make_x_y_grids(max_coord, grid_res, num_robots)
        z_grid = torch.zeros_like(x_grid)
        self.world_config = WorldConfig(
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=z_grid,
            grid_res=grid_res,
            max_coord=max_coord,
        )
        self.physics_config = PhysicsEngineConfig(num_robots=num_robots)
        for cfg in [self.robot_config, self.world_config, self.physics_config]:
            cfg.to(self.device)

        self.gridmap_frame = None
        self.robot_frame = self.get_parameter('robot_frame').value
        self.n_sim_steps = int(self.get_parameter('traj_sim_time').value / self.physics_config.dt)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.path_cost_min = np.inf
        self.path_cost_max = -np.inf
        self.pose_step = int(0.5 / self.physics_config.dt)  # publish poses with 0.5 [sec] step

        self.sampled_paths_pub = self.create_publisher(MarkerArray, '/sampled_paths', 1)
        self.path_costs_pub = self.create_publisher(Float32MultiArray, '/path_costs', 1)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.controls = self.init_controls()
        self.phys_engine = DPhysicsEngine(config=self.physics_config, robot_model=self.robot_config, device=self.device)
        self.compile_pysics_engine()

        # grid map subscriber to publish grid map tf
        gridmap_topic = self.get_parameter('gridmap_topic').value
        self._logger.info(f'Subscribing to grid map topic: {gridmap_topic}')
        self.gridmap_tf_sub = self.create_subscription(GridMap, gridmap_topic, self.gridmap_tf_callback, 1)
        self.gridmap_sub = self.create_subscription(GridMap, gridmap_topic, self.gridmap_callback, 1)
    
    def spin(self):
        try:
            rclpy.spin(self)
        except (KeyboardInterrupt, ExternalShutdownException):
            self.get_logger().info('Keyboard interrupt, shutting down...')
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def compile_pysics_engine(self):
        # compile_opts = {"max-autotune": True, "triton.cudagraphs": True, "coordinate_descent_tuning": True}
        # self.phys_engine = torch.compile(self.phys_engine, options=compile_opts)
        # init_state = PhysicsState.dummy(batch_size=self.physics_config.num_robots, robot_model=self.robot_config)
        # _ = self.phys_engine(init_state, self.controls[0], self.world_config)
        self.phys_engine = torch.compile(self.phys_engine)
    
    def gridmap_tf_callback(self, gridmap_msg):
        # publish grid map pose tf frame: create tf message from grid map pose
        grid_map_tf = TransformStamped()
        grid_map_tf.header.stamp = gridmap_msg.header.stamp
        grid_map_tf.header.frame_id = gridmap_msg.header.frame_id
        grid_map_tf.child_frame_id = 'grid_map_link'
        grid_map_tf.transform.translation.x = gridmap_msg.info.pose.position.x
        grid_map_tf.transform.translation.y = gridmap_msg.info.pose.position.y
        grid_map_tf.transform.translation.z = gridmap_msg.info.pose.position.z
        grid_map_tf.transform.rotation = gridmap_msg.info.pose.orientation
        self.tf_broadcaster.sendTransform(grid_map_tf)

    def init_controls(self):
        speed = 1. * torch.ones(self.physics_config.num_robots, device=self.device)  # m/s forward
        speed[::2] = -speed[::2]
        omega = torch.linspace(-1., 1., self.physics_config.num_robots).to(self.device)  # rad/s yaw
        flipper_vs = self.robot_config.vw_to_vels(speed, omega)
        flipper_ws = torch.zeros_like(flipper_vs)
        controls = torch.cat((flipper_vs, flipper_ws), dim=-1).to(self.device).repeat(self.n_sim_steps, 1, 1).permute(1, 0, 2)
        return controls

    def get_transform(self, from_frame, to_frame, time=None):
        """Retrieve a transformation matrix between two frames using TF2."""
        if time is None:
            time = rclpy.time.Time()
        timeout = rclpy.time.Duration(seconds=1.0)
        try:
            tf = self.tf_buffer.lookup_transform(to_frame, from_frame,
                                                 time=time, timeout=timeout)
        except Exception as ex:
            tf = self.tf_buffer.lookup_transform(to_frame, from_frame,
                                                 time=rclpy.time.Time(), timeout=timeout)
            self._logger.warning(
                f"Could not find transform from {from_frame} to {to_frame} at time {time}, using latest available transform: {ex}"
            )
        # Convert TF2 transform message to a 4x4 transformation matrix
        translation = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
        qaut = [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
        T = np.eye(4)
        R = Rotation.from_quat(qaut).as_matrix()
        T[:3, 3] = translation
        T[:3, :3] = R
        return T

    def publish_paths_and_costs(self, poses, path_costs, frame, stamp=None, pose_step=1):
        assert len(poses) == len(path_costs), 'xyz_q_np: %s, path_costs: %s' % (str(poses.shape), str(path_costs.shape))
        if stamp is None:
            stamp = rclpy.time.Time()

        num_trajs = len(poses)

        # paths marker array
        marker_array = MarkerArray()
        red = np.array([1., 0., 0.], dtype=np.float32)
        green = np.array([0., 1., 0.], dtype=np.float32)
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

        # publish path costs
        path_costs_msg = Float32MultiArray()
        path_costs_msg.data = path_costs
        self.path_costs_pub.publish(path_costs_msg)

    def gridmap_callback(self, gridmap_msg):
        # if a message is stale, do not process it
        t_now = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec / 1e9
        t_msg = gridmap_msg.header.stamp.sec + gridmap_msg.header.stamp.nanosec / 1e9
        dt = abs(t_now - t_msg)
        if dt > self.get_parameter('max_age').value:
            self._logger.warning(f'GridMap message is stale (time diff: {dt:.3f} s), skipping...')
            return

        t0 = time()
        if self.gridmap_frame is None:
            self.gridmap_frame = gridmap_msg.header.frame_id

        # convert grid map to height map
        gridmap_layer = self.get_parameter('gridmap_layer').value
        grid_map = gridmap_msg_to_numpy(gridmap_msg, gridmap_layer)
        assert grid_map.ndim == 2, 'Height map must be 2D'
        self._logger.debug('Received height map of shape: %s' % str(grid_map.shape))

        grid_map_pose = pose_to_matrix(gridmap_msg.info.pose)
        robot_pose = self.get_transform(from_frame=self.gridmap_frame, to_frame=self.robot_frame,
                                        time=gridmap_msg.header.stamp)
        if robot_pose is None:
            self._logger.warning('Could not get robot pose')
            return
        robot_pose_wrt_gridmap = np.linalg.inv(grid_map_pose) @ robot_pose
        robot_xyz_q_wrt_gridmap = pose_to_xyz_q(robot_pose_wrt_gridmap)
        t1 = time()
        self._logger.debug('Grid map preprocessing took %.3f [sec]' % (t1 - t0))

        # predict path
        grid_maps = np.repeat(grid_map[None], self.physics_config.num_robots, axis=0)
        xyz_qs_init = np.repeat(robot_xyz_q_wrt_gridmap[None], self.physics_config.num_robots, axis=0)
        t2 = time()
        xyz_qs, path_costs = self.predict_paths(grid_maps, xyz_qs_init)
        t3 = time()
        self._logger.info('Path prediction took %.3f [sec]' % (t3 - t2))

        # update path cost bounds
        if path_costs is not None:
            self.path_cost_min = min(self.path_cost_min, torch.min(path_costs).item())
            self.path_cost_max = max(self.path_cost_max, torch.max(path_costs).item())
            self._logger.debug('Path cost min: %.3f' % self.path_cost_min)
            self._logger.debug('Path cost max: %.3f' % self.path_cost_max)

        # publish paths
        if xyz_qs is not None:
            t4 = time()
            xyz_qs_np = xyz_qs.cpu().numpy()
            path_costs_np = path_costs.cpu().numpy()
            self.publish_paths_and_costs(xyz_qs_np, path_costs_np, stamp=gridmap_msg.header.stamp,
                                         frame=self.robot_frame, pose_step=self.pose_step)
            self._logger.debug('Paths publishing took %.3f [sec]' % (time() - t4))

    @torch.inference_mode()
    def predict_paths(self, grid_maps, xyz_qs_init):
        assert len(grid_maps) == len(xyz_qs_init) == self.physics_config.num_robots
        grid_maps = torch.as_tensor(grid_maps, dtype=torch.float32, device=self.device)
        assert grid_maps.shape[0] == self.physics_config.num_robots
        assert self.controls.shape == (self.physics_config.num_robots, self.n_sim_steps, 8), \
            f'controls shape: {self.controls.shape} != {(self.physics_config.num_robots, self.n_sim_steps, 8)}'

        # initial state
        x0 = torch.as_tensor(xyz_qs_init[:, :3], dtype=torch.float32, device=self.device)
        xd0 = torch.zeros_like(x0)
        q0 = torch.as_tensor(xyz_qs_init[:, 3:7], dtype=torch.float32, device=self.device)
        omega0 = torch.zeros_like(x0)
        thetas0 = torch.zeros(self.physics_config.num_robots, self.robot_config.num_driving_parts).to(self.device)
        state0 = PhysicsState(x0, xd0, q0, omega0, thetas0)

        # simulate trajectories
        states = deque(maxlen=self.n_sim_steps)
        auxs = deque(maxlen=self.n_sim_steps)
        state = state0
        # update grid maps
        self.world_config.z_grid = grid_maps
        for i in range(self.n_sim_steps):
            state, der, aux = self.phys_engine(state, self.controls[:, i], self.world_config)
            states.append(state)
            auxs.append(aux)
        states = vectorize_states(states)
        auxs = vectorize_states(auxs)

        # path poses from states
        xyz = states.x.permute(1, 0, 2)  # (num_robots, n_sim_steps, 3)
        q = states.q.permute(1, 0, 2)  # (num_robots, n_sim_steps, 4)
        xyz_q = torch.cat((xyz, q), dim=-1)
        assert xyz_q.shape == (self.physics_config.num_robots, self.n_sim_steps, 7)

        # compute path costs
        F_springs, F_frictions = auxs.F_spring, auxs.F_friction
        n_robot_points = self.robot_config.num_driving_parts * self.robot_config.points_per_driving_part + self.robot_config.points_per_body
        assert F_springs.shape == (self.n_sim_steps, self.physics_config.num_robots, n_robot_points, 3)
        assert F_frictions.shape == (self.n_sim_steps, self.physics_config.num_robots, n_robot_points, 3)
        path_costs = F_springs.norm(dim=-1).mean(dim=-1).std(dim=0)
        assert xyz_q.shape == (self.physics_config.num_robots, self.n_sim_steps, 7), f'xyz_q shape: {xyz_q.shape}'
        assert path_costs.shape == (self.physics_config.num_robots,)

        return xyz_q, path_costs


def main(args=None):
    rclpy.init(args=args)
    node = DiffPhysEngineNode()
    node.spin()


if __name__ == '__main__':
    main()