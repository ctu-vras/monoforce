#!/usr/bin/env python

from collections import deque
from time import time
import torch
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from monoforce.models.physics_engine.engine.engine import DPhysicsEngine, PhysicsState
from monoforce.monoforce.config import WorldConfig, RobotModelConfig, PhysicsEngineConfig
from monoforce.models.physics_engine.engine.engine_state import vectorize_iter_of_states as vectorize_states
from monoforce.ros import poses_to_marker, poses_to_path, gridmap_msg_to_numpy
from monoforce.transformations import pose_to_xyz_q
from nav_msgs.msg import Path
from grid_map_msgs.msg import GridMap
from ros_numpy import numpify
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray
from tf2_ros import TransformBroadcaster


class DiffPhysEngineNode:
    def __init__(self,
                 robot_config: RobotModelConfig,
                 world_config: WorldConfig,
                 physics_config: PhysicsEngineConfig,
                 gridmap_topic='/grid_map/terrain',
                 gridmap_layer='elevation',
                 robot_frame='base_link',
                 max_age=0.5,
                 device='cpu',
                 traj_sim_time=5.0):
        self.robot_frame = robot_frame
        self.robot_config = robot_config.to(device)
        self.world_config = world_config.to(device)
        self.physics_cfg = physics_config.to(device)
        self.gridmap_layer = gridmap_layer
        self.gridmap_frame = None
        self.dt = self.physics_cfg.dt
        self.n_sim_steps = int(traj_sim_time / self.dt)
        self.max_age = max_age
        self.device = device
        self.num_robots = self.physics_cfg.num_robots

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

        self.dphysics = DPhysicsEngine(config=self.physics_cfg, robot_model=self.robot_config, device=self.device)
        self.controls = self.init_controls()
        # grid map subscriber
        self.gridmap_sub = rospy.Subscriber(gridmap_topic, GridMap, self.gridmap_callback)

    def gridmap_tf_callback(self, gridmap_msg):
        # publish grid map pose tf frame: create tf message from grid map pose
        grid_map_tf = TransformStamped()
        grid_map_tf.header.stamp = gridmap_msg.info.header.stamp
        grid_map_tf.header.frame_id = gridmap_msg.info.header.frame_id
        grid_map_tf.child_frame_id = 'grid_map_link'
        grid_map_tf.transform.translation.x = gridmap_msg.info.pose.position.x
        grid_map_tf.transform.translation.y = gridmap_msg.info.pose.position.y
        grid_map_tf.transform.translation.z = gridmap_msg.info.pose.position.z
        grid_map_tf.transform.rotation = gridmap_msg.info.pose.orientation
        self.tf_broadcaster.sendTransform(grid_map_tf)

    def init_controls(self):
        speed = 1. * torch.ones(self.num_robots, device=self.device)  # m/s forward
        speed[::2] = -speed[::2]
        omega = torch.linspace(-1., 1., self.num_robots).to(self.device)  # rad/s yaw
        controls = self.robot_config.vw_to_vels(speed, omega)
        flipper_controls = torch.zeros_like(controls)
        controls_all = torch.cat((controls, flipper_controls), dim=-1).to(self.device).repeat(self.n_sim_steps, 1, 1).permute(1, 0, 2)
        return controls_all

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
            # rospy.logwarn(f'Stale grid map message received ({dt.to_sec():.1f} > {self.max_age} [sec]), skipping')
            return

        t0 = time()
        if self.gridmap_frame is None:
            self.gridmap_frame = gridmap_msg.info.header.frame_id

        # convert grid map to height map
        grid_map = gridmap_msg_to_numpy(gridmap_msg, self.gridmap_layer)
        # assert not np.all(np.isnan(grid_map)) and np.all(np.isfinite(grid_map))
        assert grid_map.ndim == 2, 'Height map must be 2D'
        rospy.logdebug('Received height map of shape: %s' % str(grid_map.shape))

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
        grid_maps = np.repeat(grid_map[None], self.num_robots, axis=0)
        xyz_qs_init = np.repeat(robot_xyz_q_wrt_gridmap[None], self.num_robots, axis=0)
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
                                         frame=self.robot_frame, pose_step=self.pose_step)
            rospy.logdebug('Paths publishing took %.3f [sec]' % (time() - t4))

    @staticmethod
    def spin():
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            pass

    @torch.inference_mode()
    def predict_paths(self, grid_maps, xyz_qs_init):
        assert len(grid_maps) == len(xyz_qs_init) == self.num_robots
        grid_maps = torch.as_tensor(grid_maps, dtype=torch.float32, device=self.device)
        assert grid_maps.shape[0] == self.num_robots
        assert self.controls.shape == (self.num_robots, self.n_sim_steps, 8), \
            f'controls shape: {self.controls.shape} != {(self.num_robots, self.n_sim_steps, 8)}'

        # initial state
        x0 = torch.as_tensor(xyz_qs_init[:, :3], dtype=torch.float32, device=self.device)
        xd0 = torch.zeros_like(x0)
        q0 = torch.as_tensor(xyz_qs_init[:, 3:7], dtype=torch.float32, device=self.device)
        omega0 = torch.zeros_like(x0)
        thetas0 = torch.zeros(self.num_robots, self.robot_config.num_driving_parts).to(self.device)
        state0 = PhysicsState(x0, xd0, q0, omega0, thetas0)

        # simulate trajectories
        states = deque(maxlen=self.n_sim_steps)
        auxs = deque(maxlen=self.n_sim_steps)
        state = state0
        # update grid maps
        self.world_config.z_grid = grid_maps
        for i in range(self.n_sim_steps):
            state, der, aux = self.dphysics(state, self.controls[:, i], self.world_config)
            states.append(state)
            auxs.append(aux)
        states = vectorize_states(states)
        auxs = vectorize_states(auxs)

        # path poses from states
        xyz = states.x.permute(1, 0, 2)  # (num_robots, n_sim_steps, 3)
        q = states.q.permute(1, 0, 2)  # (num_robots, n_sim_steps, 4)
        xyz_q = torch.cat((xyz, q), dim=-1)
        assert xyz_q.shape == (self.num_robots, self.n_sim_steps, 7)

        # compute path costs
        F_springs, F_frictions = auxs.F_spring, auxs.F_friction
        n_robot_points = self.robot_config.num_driving_parts * self.robot_config.points_per_driving_part + self.robot_config.points_per_body
        assert F_springs.shape == (self.n_sim_steps, self.num_robots, n_robot_points, 3)
        assert F_frictions.shape == (self.n_sim_steps, self.num_robots, n_robot_points, 3)
        path_costs = F_springs.norm(dim=-1).mean(dim=-1).std(dim=0)
        assert xyz_q.shape == (self.num_robots, self.n_sim_steps, 7), f'xyz_q shape: {xyz_q.shape}'
        assert path_costs.shape == (self.num_robots,)

        return xyz_q, path_costs


def main():
    rospy.init_node('diff_physics', anonymous=True, log_level=rospy.DEBUG)

    robot = rospy.get_param('~robot', 'marv')
    device = rospy.get_param('~device', 'cuda' if torch.cuda.is_available() else 'cpu')
    robot_config = RobotModelConfig(kind=robot)
    grid_res = 0.1  # 10cm per grid cell
    max_coord = 6.4  # meters
    num_robots = rospy.get_param('~num_robots', 32)
    DIM = int(2 * max_coord / grid_res)
    xint = torch.linspace(-max_coord, max_coord, DIM)
    yint = torch.linspace(-max_coord, max_coord, DIM)
    x_grid, y_grid = torch.meshgrid(xint, yint, indexing="xy")
    z_grid = torch.zeros_like(x_grid)
    world_config = WorldConfig(
        x_grid=x_grid.repeat(num_robots, 1, 1),
        y_grid=y_grid.repeat(num_robots, 1, 1),
        z_grid=z_grid.repeat(num_robots, 1, 1),
        grid_res=grid_res,
        max_coord=max_coord,
    )
    physics_config = PhysicsEngineConfig(num_robots=num_robots)
    robot_frame = rospy.get_param('~robot_frame', 'base_link')
    gridmap_topic = rospy.get_param('~gridmap_topic')
    gridmap_layer = rospy.get_param('~gridmap_layer', 'elevation')
    max_age = rospy.get_param('~max_age', 0.5)

    node = DiffPhysEngineNode(robot_config=robot_config,
                              world_config=world_config,
                              physics_config=physics_config,
                              robot_frame=robot_frame,
                              gridmap_topic=gridmap_topic,
                              gridmap_layer=gridmap_layer,
                              max_age=max_age,
                              device=device)
    node.spin()


if __name__ == '__main__':
    main()
