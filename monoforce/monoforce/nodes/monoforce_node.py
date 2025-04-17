#!/usr/bin/env python

import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from collections import deque
from time import time

import rclpy
from rclpy.impl.logging_severity import LoggingSeverity

from sensor_msgs.msg import CameraInfo, CompressedImage
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray

import tf2_ros
from geometry_msgs.msg import TransformStamped
from grid_map_msgs.msg import GridMap

from monoforce.ros import terrain_to_gridmap_msg, poses_to_path, poses_to_marker
from monoforce.models.physics_engine.engine.engine import DPhysicsEngine, PhysicsState
from monoforce.configs import WorldConfig, RobotModelConfig, PhysicsEngineConfig
from monoforce.models.physics_engine.utils.environment import make_x_y_grids
from monoforce.models.physics_engine.engine.engine_state import vectorize_iter_of_states as vectorize_states

from .terrain_encoder import TerrainEncoder


class MonoForce(TerrainEncoder):
    """
    MonoForce node for predicting terrain properties and robot's trajectories from RGB images.
    """
    def __init__(self):
        super(MonoForce, self).__init__()
        self.declare_parameter('num_robots', 32)
        self.declare_parameter('traj_sim_time', 5.0)
        self.declare_parameter('grid_res', 0.1)
        self.declare_parameter('max_coord', 6.4)
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
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.controls = self.init_controls()
        self.phys_engine = DPhysicsEngine(config=self.physics_config, robot_model=self.robot_config, device=self.device)
        self.compile_pysics_engine()

    def init_controls(self):
        speed = 1. * torch.ones(self.physics_config.num_robots, device=self.device)  # m/s forward
        speed[::2] = -speed[::2]
        omega = torch.linspace(-1., 1., self.physics_config.num_robots).to(self.device)  # rad/s yaw
        flipper_vs = self.robot_config.vw_to_vels(speed, omega)
        flipper_ws = torch.zeros_like(flipper_vs)
        controls = torch.cat((flipper_vs, flipper_ws), dim=-1).to(self.device).repeat(self.n_sim_steps, 1, 1)
        controls = controls.permute(1, 0, 2)  # (N, T, D)
        return controls

    def compile_pysics_engine(self):
        self.phys_engine = torch.compile(self.phys_engine)
        self._logger.info('PyTorch Physics Engine is compiled')

    def predict_paths(self, grid_maps, xyz_qs_init=None):
        N, T = self.physics_config.num_robots, self.n_sim_steps
        if xyz_qs_init is None:
            xyz_qs_init = torch.zeros(N, 7, device=self.device)
            xyz_qs_init[:, 3] = 1.0  # set initial quaternion to identity, x, y, z, qw, qx, qy, qz
        assert len(grid_maps) == len(xyz_qs_init) == N
        grid_maps = torch.as_tensor(grid_maps, dtype=torch.float32, device=self.device)
        assert grid_maps.shape[0] == N
        assert self.controls.shape == (N, T, 8), f'controls shape: {self.controls.shape} != {(N, T, 8)}'

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

        # path poses from states
        xyz = states.x.permute(1, 0, 2)  # (num_robots, n_sim_steps, 3)
        q = states.q.permute(1, 0, 2)  # (num_robots, n_sim_steps, 4)
        xyzq = torch.cat((xyz, q), dim=-1)
        assert xyzq.shape == (N, T, 7), f'xyzq shape: {xyzq.shape}'

        # compute path costs
        omegas = states.omega.permute(1, 0, 2)  # (num_robots, n_sim_steps, 3)
        assert omegas.shape == (N, T, 3), f'omegas shape: {omegas.shape}'
        path_costs = omegas.norm(dim=-1).mean(dim=-1)
        assert path_costs.shape == (N,)

        return xyzq, path_costs

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

    @torch.inference_mode()
    def proc(self, *msgs):
        n = len(msgs)
        assert n % 2 == 0
        for i in range(n // 2):
            assert isinstance(msgs[i], CompressedImage), 'First %d messages must be Image' % (n // 2)
            assert isinstance(msgs[i + n // 2], CameraInfo), 'Last %d messages must be CameraInfo' % (n // 2)
            assert msgs[i].header.frame_id == msgs[i + n // 2].header.frame_id, \
                'Image and CameraInfo messages must have the same frame_id'
        # preprocessing
        img_msgs = msgs[:n // 2]
        info_msgs = msgs[n // 2:]
        inputs = self.get_lss_inputs(img_msgs, info_msgs)
        inputs = [i.to(self.device) for i in inputs]

        # model inference
        terrain = self.terrain_encoder(*inputs)
        elevation, friction = terrain['terrain'], terrain['friction']
        self._logger.info('Predicted height map shape: %s' % str(elevation.shape))

        # publish terrain as grid map
        stamp = msgs[0].header.stamp
        elevation = elevation.squeeze().cpu().numpy()
        friction = friction.squeeze().cpu().numpy()
        gridmap_msg = terrain_to_gridmap_msg(layers=[elevation, friction], layer_names=['elevation', 'friction'],
                                          grid_res=self.world_config.grid_res)
        gridmap_msg.header.stamp = stamp
        gridmap_msg.header.frame_id = self.robot_frame
        self.gridmap_pub.publish(gridmap_msg)

        # predict path
        elevations = np.repeat(elevation[None], self.physics_config.num_robots, axis=0)
        t0 = time()
        xyz_qs, path_costs = self.predict_paths(elevations)
        self._logger.info('Predicted paths time: %.3f sec' % (time() - t0))
        self._logger.debug('Predicted paths shape: %s' % str(xyz_qs.shape))
        self._logger.debug('Predicted path costs shape: %s' % str(path_costs.shape))

        # update path cost bounds
        if path_costs is not None:
            self.path_cost_min = min(self.path_cost_min, torch.min(path_costs).item())
            self.path_cost_max = max(self.path_cost_max, torch.max(path_costs).item())
            self._logger.debug('Path cost min: %.3f' % self.path_cost_min)
            self._logger.debug('Path cost max: %.3f' % self.path_cost_max)

        # publish paths
        if xyz_qs is not None:
            t1 = time()
            xyz_qs_np = xyz_qs.cpu().numpy()
            path_costs_np = path_costs.cpu().numpy()
            self.publish_paths_and_costs(xyz_qs_np, path_costs_np, stamp=gridmap_msg.header.stamp,
                                         frame=self.robot_frame, pose_step=self.pose_step)
            self._logger.debug('Paths publishing took %.3f [sec]' % (time() - t1))


def main(args=None):
    rclpy.init(args=args)
    node = MonoForce()
    node.start()
    node.spin()


if __name__ == '__main__':
    main()