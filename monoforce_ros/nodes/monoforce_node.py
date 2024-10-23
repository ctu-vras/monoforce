#!/usr/bin/env python

import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import rospy
from grid_map_msgs.msg import GridMap
from sensor_msgs.msg import CameraInfo, CompressedImage
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray
from monoforce.ros import height_map_to_gridmap_msg, poses_to_path, poses_to_marker
from monoforce.utils import read_yaml, timing
from monoforce.dphys_config import DPhysConfig
from monoforce.models.dphysics import DPhysics, vw_to_tracks_vel
from terrain_encoder import TerrainEncoder
import rospkg


class MonoForce(TerrainEncoder):
    """
    MonoForce node for predicting terrain properties and robot's trajectories from RGB images.
    """
    def __init__(self, lss_cfg):
        super(MonoForce, self).__init__(lss_cfg)
        # differentiable physics configs
        self.robot = rospy.get_param('~robot', 'robot')
        self.allow_backward = rospy.get_param('~allow_backward', True)
        self.dphys_cfg = DPhysConfig(robot=self.robot)
        self.dphysics = DPhysics(self.dphys_cfg, device=self.device)
        self.track_vels = self.init_controls(self.dphys_cfg.vel_max,
                                             self.dphys_cfg.omega_max,
                                             robot_size=self.dphys_cfg.robot_size,
                                             allow_backward=self.allow_backward)
        rospy.loginfo('Control inputs are set up. Shape: %s' % str(self.track_vels.shape))
        self.path_cost_min = np.inf
        self.path_cost_max = -np.inf
        self.pose_step = int(0.2 / self.dphys_cfg.dt)  # publish poses with 0.2 [sec] step

        self.sampled_paths_pub = rospy.Publisher('/sampled_paths', MarkerArray, queue_size=1)
        self.lc_path_pub = rospy.Publisher('/lower_cost_path', Path, queue_size=1)
        self.path_costs_pub = rospy.Publisher('/path_costs', Float32MultiArray, queue_size=1)
        rospy.loginfo('MonoForce node is ready')

    def init_controls(self, linear_vel, angular_vel_max, robot_size, allow_backward=True):
        """
        Initialize control commands through input linear and angular velocities.
        The differentiable drive robot model is used to calculate left and right 'track' (wheel) speed:
        v_l(t) = v(t) + r * w(t)
        v_r(t) = v(t) - r * w(t),
        r is half of the robot's width, r = robot_size[1] / 2.

        where v(t) = V = const for t in [0 .. T], linear velocity,
        w(t) = [-W_max .. W_max], angular velocity.,
        T is trajectories time horizon.
        """
        N_trajs = self.dphys_cfg.n_sim_trajs
        N_steps = int(self.dphys_cfg.traj_sim_time / self.dphys_cfg.dt)
        assert self.dphys_cfg.n_sim_trajs % 2 == 0, 'n_sim_trajs must be even'
        if allow_backward:
            vels = torch.cat([-linear_vel * torch.ones((N_trajs // 2, N_steps)),
                              +linear_vel * torch.ones((N_trajs // 2, N_steps))])
            omegas = torch.cat([torch.linspace(-angular_vel_max, angular_vel_max, N_trajs // 2),
                                torch.linspace(-angular_vel_max, angular_vel_max, N_trajs // 2)]).repeat(N_steps, 1).T
        else:
            vels = linear_vel * torch.ones((N_trajs, N_steps))
            omegas = torch.linspace(-angular_vel_max, angular_vel_max, N_trajs).repeat(N_steps, 1).T
        assert vels.shape == (N_trajs, N_steps), 'vels shape: %s' % str(vels.shape)
        assert omegas.shape == (N_trajs, N_steps), 'omegas shape: %s' % str(omegas.shape)

        # convert linear and angular velocities to track velocities
        n_tracks = len(self.dphys_cfg.driving_parts)
        track_vels = vw_to_tracks_vel(vels, omegas, robot_size, n_tracks=n_tracks)
        track_vels = torch.stack(track_vels, dim=2)
        assert track_vels.shape == (N_trajs, N_steps, n_tracks), 'track_vels shape: %s' % str(track_vels.shape)

        return track_vels
    
    def predict_paths(self, grid_maps, xyz_qs_init, frictions):
        assert len(grid_maps) == len(xyz_qs_init) == self.dphys_cfg.n_sim_trajs
        # for grid_map in grid_maps:
        #     assert isinstance(grid_map, np.ndarray)
        #     assert grid_map.shape[0] == grid_map.shape[1]
        grid_maps = torch.as_tensor(grid_maps, dtype=torch.float32, device=self.device)
        assert grid_maps.shape[0] == self.dphys_cfg.n_sim_trajs
        controls = torch.as_tensor(self.track_vels, dtype=torch.float32, device=self.device)
        n_tracks = len(self.dphys_cfg.driving_parts)
        n_sim_steps = int(self.dphys_cfg.traj_sim_time / self.dphys_cfg.dt)
        assert controls.shape == (self.dphys_cfg.n_sim_trajs, n_sim_steps, n_tracks)

        # initial state
        x = torch.as_tensor(xyz_qs_init[:, :3], dtype=torch.float32, device=self.device)
        xd = torch.zeros_like(x)
        R = torch.as_tensor(Rotation.from_quat(xyz_qs_init[:, 3:]).as_matrix(), dtype=torch.float32, device=self.device)
        R.repeat(x.shape[0], 1, 1)
        omega = torch.zeros_like(x)
        x_points = torch.as_tensor(self.dphys_cfg.robot_points, dtype=torch.float32, device=self.device).repeat(x.shape[0], 1, 1)
        x_points = x_points @ R.transpose(1, 2) + x.unsqueeze(1)
        state0 = (x, xd, R, omega, x_points)

        # simulate trajectories
        states, forces = self.dphysics(grid_maps, controls=controls, state=state0, friction=frictions)
        Xs, Xds, Rs, Omegas, X_points = states
        assert Xs.shape == (self.dphys_cfg.n_sim_trajs, n_sim_steps, 3)
        assert Rs.shape == (self.dphys_cfg.n_sim_trajs, n_sim_steps, 3, 3)

        # convert rotation matrices to quaternions
        poses = torch.zeros((self.dphys_cfg.n_sim_trajs, n_sim_steps, 4, 4), device=self.device)
        poses[:, :, :3, 3] = Xs
        poses[:, :, :3, :3] = Rs
        poses[:, :, 3, 3] = 1.0
        assert not torch.any(torch.isnan(poses))

        # compute path costs
        F_springs, F_frictions = forces
        assert F_springs.shape == (self.dphys_cfg.n_sim_trajs, n_sim_steps, len(self.dphys_cfg.robot_points), 3)
        assert F_frictions.shape == (self.dphys_cfg.n_sim_trajs, n_sim_steps, len(self.dphys_cfg.robot_points), 3)
        path_costs = torch.norm(F_springs, dim=-1).std(dim=-1).std(dim=-1)
        assert not torch.any(torch.isnan(path_costs))
        assert poses.shape == (self.dphys_cfg.n_sim_trajs, n_sim_steps, 4, 4)
        assert path_costs.shape == (self.dphys_cfg.n_sim_trajs,)

        return poses, path_costs

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
        path_msg = poses_to_path(lower_cost_xyz_q[::pose_step], stamp=stamp, frame_id=self.robot_frame)
        self.lc_path_pub.publish(path_msg)

        # publish path costs
        path_costs_msg = Float32MultiArray()
        path_costs_msg.data = path_costs
        self.path_costs_pub.publish(path_costs_msg)

    def proc(self, *msgs):
        n = len(msgs)
        assert n % 2 == 0
        for i in range(n // 2):
            assert isinstance(msgs[i], CompressedImage), 'First %d messages must be CompressedImage' % (n // 2)
            assert isinstance(msgs[i + n // 2], CameraInfo), 'Last %d messages must be CameraInfo' % (n // 2)
            assert msgs[i].header.frame_id == msgs[i + n // 2].header.frame_id, \
                'Image and CameraInfo messages must have the same frame_id'
        img_msgs = msgs[:n // 2]
        info_msgs = msgs[n // 2:]
        inputs = self.get_lss_inputs(img_msgs, info_msgs)
        inputs = [i.to(self.device) for i in inputs]
        out = self.model(*inputs)
        height_terrain, friction = out['terrain'], out['friction']
        rospy.loginfo('Predicted height map shape: %s' % str(height_terrain.shape))

        grid_maps = height_terrain.squeeze(1).repeat(self.dphys_cfg.n_sim_trajs, 1, 1)
        frictions = friction.squeeze(1).repeat(self.dphys_cfg.n_sim_trajs, 1, 1)
        xyz_qs_init = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]]).repeat(self.dphys_cfg.n_sim_trajs, 1)
        with torch.no_grad():
            xyz_qs, path_costs = self.predict_paths(grid_maps, xyz_qs_init, frictions)

        # update path cost bounds
        if path_costs is not None:
            self.path_cost_min = min(self.path_cost_min, torch.min(path_costs).item())
            self.path_cost_max = max(self.path_cost_max, torch.max(path_costs).item())
            rospy.logdebug('Path cost min: %.3f' % self.path_cost_min)
            rospy.logdebug('Path cost max: %.3f' % self.path_cost_max)

        # publish paths
        stamp = msgs[0].header.stamp
        if xyz_qs is not None:
            xyz_qs_np = xyz_qs.cpu().numpy()
            path_costs_np = path_costs.cpu().numpy()
            self.publish_paths_and_costs(xyz_qs_np, path_costs_np, stamp=stamp,
                                         frame=self.robot_frame, pose_step=self.pose_step)
            rospy.loginfo(f'Published paths (shape: {xyz_qs_np.shape}) and path costs (shape: {path_costs_np.shape})')

        # publish height map as grid map
        grid_msg = height_map_to_gridmap_msg(height=height_terrain.squeeze().cpu().numpy(),
                                             mask=friction.squeeze().cpu().numpy(),
                                             grid_res=self.lss_cfg['grid_conf']['xbound'][2],
                                             xyz=np.array([0., 0., 0.]),
                                             q=np.array([0., 0., 0., 1.]))
        grid_msg.info.header.stamp = stamp
        grid_msg.info.header.frame_id = self.robot_frame
        self.gridmap_pub.publish(grid_msg)


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
