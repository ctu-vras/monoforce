import numpy as np
import warp as wp
import warp.sim.render
from scipy.spatial.transform import Rotation as R
import sys
np.set_printoptions(threshold=sys.maxsize)


def get_heightmap_vis_ids(shp):
    # create triangles from the points in the grid
    heightmap_vis_indices = []
    for i in range(shp[0] - 1):
        for j in range(shp[1] - 1):
            heightmap_vis_indices.append([i * shp[1] + j, (i + 1) * shp[1] + j, (i + 1) * shp[1] + j + 1])
            heightmap_vis_indices.append([i * shp[1] + j, (i + 1) * shp[1] + j + 1, i * shp[1] + j + 1])
    return heightmap_vis_indices


def generate_force_vis(points, forces, scale=0.001):
    force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
    line_pts = np.zeros((len(points)*2, 3))
    line_pts[::2] = [0, 0, -10]
    line_pts[1::2] = [0, 0, -11]
    indices = np.arange(len(points)*2)
    for i in range(len(points)):
        if force_norms[i] > 1e-3:
            line_pts[2*i] = points[i]
            line_pts[2*i+1] = points[i] + forces[i]*scale
    return line_pts, indices


def combine_transforms(t1, t2):
    pos1, quat1 = t1[:3], t1[3:]
    pos2, quat2 = t2[:3], t2[3:]
    pos = pos1 + R.from_quat(quat1).apply(pos2)
    quat = (R.from_quat(quat1) * R.from_quat(quat2)).as_quat()
    return np.concatenate([pos, quat])


@wp.struct
class Heightmap:
    heights: wp.array2d(dtype=wp.float32)
    ke: wp.array2d(dtype=wp.float32)
    kd: wp.array2d(dtype=wp.float32)
    kf: wp.array2d(dtype=wp.float32)
    origin: wp.vec3
    resolution: wp.float32
    width: wp.int32
    length: wp.int32

@wp.kernel
def eval_heightmap_collisions_shoot(height_map_array: wp.array(dtype=Heightmap),
                                    body_q: wp.array2d(dtype=wp.transformf),
                                    body_qd: wp.array2d(dtype=wp.spatial_vectorf),
                                    sim_idx: int,
                                    T_s: int,
                                    track_velocities: wp.array3d(dtype=wp.float32),
                                    contact_points: wp.array3d(dtype=wp.vec3),
                                    constraint_forces: wp.array2d(dtype=wp.vec3),
                                    friction_forces: wp.array2d(dtype=wp.vec3),
                                    collisions: wp.array2d(dtype=wp.vec3),
                                    body_f: wp.array2d(dtype=wp.spatial_vectorf)):
    shoot_idx, robot_idx, contact_idx = wp.tid()

    # parse heightmap data
    height_map = height_map_array[robot_idx]
    heights = height_map.heights
    kes = height_map.ke
    kds = height_map.kd
    kfs = height_map.kf
    hm_origin = height_map.origin
    hm_res = height_map.resolution
    width = height_map.width
    length = height_map.length

    # compute current shoot simulation timestep
    current_state_idx = shoot_idx * T_s + sim_idx
    current_force_idx = (T_s - 1) * shoot_idx + sim_idx

    # acess simulation state
    robot_to_world = body_q[current_state_idx, robot_idx]
    robot_to_world_speed = body_qd[current_state_idx, robot_idx]
    wheel_to_robot_pos = contact_points[shoot_idx, robot_idx, contact_idx]

    # transform contact state to heightmap frame
    forward_to_world = wp.transform_vector(robot_to_world, wp.vec3(1.0, 0.0, 0.0))
    wheel_to_world_pos = wp.transform_point(robot_to_world, wheel_to_robot_pos)
    wheel_to_world_vel = wp.cross(wp.spatial_top(robot_to_world_speed), wheel_to_robot_pos) + wp.spatial_bottom(
        robot_to_world_speed)
    wheel_to_hm = wheel_to_world_pos - hm_origin
    # x, y normalized by the heightmap resolution
    x_n = wheel_to_hm[0] / hm_res
    y_n = wheel_to_hm[1] / hm_res
    # cell index
    u = wp.int(wp.floor(x_n))
    v = wp.int(wp.floor(y_n))

    if u < 0 or u >= width or v < 0 or v >= length:  # cell outside heightmap
        return

    # hm_height = hm[x_id, z_id]  # simpler version without interpolation

    # relative position of the wheel inside the cell
    x_r = x_n - wp.float(u)
    y_r = y_n - wp.float(v)

    # useful terms for height and terrain normal
    a = heights[u, v]
    b = heights[u + 1, v]
    c = heights[u, v + 1]
    d = heights[u + 1, v + 1]

    adbc = a + d - b - c
    ba = b - a
    ca = c - a

    hm_height = x_r * y_r * adbc + x_r * ba + y_r * ca + a

    wheel_height = wheel_to_hm[2]
    # This particular wheel is above ground, so no collision force is generated
    if wheel_height > hm_height:
        return

    # compute normal to the terrain at the horizontal position of the wheel
    n = wp.vec3(-y_r * adbc - ba, -x_r * adbc - ca, 1.0)
    n = wp.normalize(n)

    # depth of penetration
    d = hm_height - wheel_height

    # normal and tangential velocity components
    v_n = wp.dot(wheel_to_world_vel, n)
    v_t = wheel_to_world_vel - n * v_n

    # compute the track velocity at the wheel position
    tangential_track_direction = forward_to_world - n*wp.dot(forward_to_world, n)
    tangential_track_velocity = wp.vec3(0.0, 0.0, 0.0)
    if wp.length(tangential_track_direction) > 1e-4:
        vel_idx = (contact_idx % 2)
        track_vel = track_velocities[current_state_idx, robot_idx, vel_idx]  # left and right track velocities
        tangential_track_velocity = wp.normalize(tangential_track_direction) * track_vel

    # compute the constraint (penetration force) and friction force
    constraint_force = n * (kes[u, v] * d - kds[u, v] * v_n)
    friction_force = -kfs[u, v] * (v_t - tangential_track_velocity) * wp.length(constraint_force)
    total_force = constraint_force + friction_force

    # combine into wrench and add to total robot force
    robot_wrench = wp.spatial_vector(wp.cross(wheel_to_robot_pos, total_force), total_force)
    wp.atomic_add(body_f, current_force_idx, robot_idx, robot_wrench)

    if robot_idx != 0:
        return

    # Store the contact info only for the first robot TODO: vis all robots?
    constraint_forces[current_state_idx, contact_idx] = constraint_force
    friction_forces[current_state_idx, contact_idx] = friction_force
    collisions[current_state_idx, contact_idx] = wp.vec3(wheel_to_world_pos[0], wheel_to_world_pos[1], hm_height + hm_origin[2])

@wp.kernel
def lossL2(gt_body_q: wp.array(dtype=wp.transformf), sim_body_q: wp.array2d(dtype=wp.transformf),
           timestamps: wp.array(dtype=wp.int32), robot_idx: int, loss: wp.array(dtype=float)):
    tid = wp.tid()
    timestamp = timestamps[tid]
    sim_robot_transform = sim_body_q[timestamp, robot_idx]
    gt_robot_transform = gt_body_q[tid]

    # get target state
    target_pos = wp.transform_get_translation(gt_robot_transform)
    target_att = wp.transform_get_rotation(gt_robot_transform)

    # get simulated state
    sim_pos = wp.transform_get_translation(sim_robot_transform)
    sim_att = wp.transform_get_rotation(sim_robot_transform)

    # compute distances and add to loss atomically
    dist = wp.length_sq(target_pos - sim_pos)
    att_err = wp.length_sq(target_att - sim_att)
    wp.atomic_add(loss, 0, dist + att_err)


@wp.kernel
def torch_hms_to_warp(torch_hms: wp.array3d(dtype=wp.float32), warp_hms: wp.array(dtype=Heightmap)):
    robot_idx, i, j = wp.tid()
    warp_hms[robot_idx].heights[i, j] = torch_hms[robot_idx, i, j]


@wp.kernel
def warp_hm_to_torch(heights: wp.array2d(dtype=wp.float32), torch_hms: wp.array3d(dtype=wp.float32), robot_idx: int):
    i, j = wp.tid()
    torch_hms[robot_idx, i, j] = heights[i, j]


@wp.kernel
def copy_state(body_q: wp.array2d(dtype=wp.transformf), state_body_q: wp.array(dtype=wp.transformf), sim_idx: int):
    """copy the simulation state body_q into rendering state state_body_q at index sim_idx"""
    tid = wp.tid()
    state_body_q[tid] = body_q[sim_idx][tid]

@wp.kernel
def continue_velocities(body_qd: wp.array2d(dtype=wp.spatial_vector), T_s: int):
    shoot_idx, robot_idx = wp.tid()
    body_qd[(shoot_idx + 1) * T_s, robot_idx] = body_qd[(shoot_idx + 1) * T_s - 1, robot_idx]

@wp.kernel
def step_init_vels(body_qd: wp.array2d(dtype=wp.spatial_vector), T_s: int, step_size: float):
    shoot_idx, robot_idx = wp.tid()
    init_vel = body_qd[(shoot_idx + 1) * T_s, robot_idx]
    prev_vel = body_qd[(shoot_idx + 1) * T_s - 1, robot_idx]
    body_qd[(shoot_idx + 1) * T_s, robot_idx] = init_vel + (prev_vel - init_vel) * step_size

@wp.kernel
def init_shoot_poses(body_q: wp.array2d(dtype=wp.transform), shoot_init_poses: wp.array2d(dtype=wp.transform), T_s: int):
    shoot_idx, robot_idx = wp.tid()
    body_q[shoot_idx * T_s, robot_idx] = shoot_init_poses[shoot_idx, robot_idx]

@wp.kernel
def init_shoot_vels(body_qd: wp.array2d(dtype=wp.spatial_vectorf), shoot_init_vels: wp.array2d(dtype=wp.spatial_vectorf), T_s: int):
    shoot_idx, robot_idx = wp.tid()
    body_qd[shoot_idx * T_s, robot_idx] = shoot_init_vels[shoot_idx, robot_idx]

@wp.kernel
def save_shoot_init_vels(body_qd: wp.array2d(dtype=wp.spatial_vectorf), shoot_init_vels: wp.array2d(dtype=wp.spatial_vectorf), T_s: int):
    shoot_idx, robot_idx = wp.tid()
    shoot_init_vels[shoot_idx, robot_idx] = body_qd[shoot_idx * T_s, robot_idx]

@wp.kernel
def integrate_bodies_shoot(
    body_q: wp.array2d(dtype=wp.transform),
    body_qd: wp.array2d(dtype=wp.spatial_vector),
    sim_idx: int,
    T_s: int,
    body_f: wp.array2d(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    I: wp.array(dtype=wp.mat33),
    inv_m: wp.array(dtype=float),
    inv_I: wp.array(dtype=wp.mat33),
    gravity: wp.vec3,
    angular_damping: float,
    dt: float,
):
    shoot_idx, robot_idx = wp.tid()

    state_timestep = shoot_idx * T_s + sim_idx
    force_timestep = shoot_idx * (T_s - 1) + sim_idx

    # positions
    q = body_q[state_timestep, robot_idx]
    qd = body_qd[state_timestep, robot_idx]
    f = body_f[force_timestep, robot_idx]

    # masses
    mass = m[0]
    inv_mass = inv_m[0]  # 1 / mass

    inertia = I[0]
    inv_inertia = inv_I[0]  # inverse of 3x3 inertia matrix

    # unpack transform
    x0 = wp.transform_get_translation(q)
    r0 = wp.transform_get_rotation(q)

    # unpack spatial twist
    w0 = wp.spatial_top(qd)
    v0 = wp.spatial_bottom(qd)

    # unpack spatial wrench
    t0 = wp.spatial_top(f)
    f0 = wp.spatial_bottom(f)

    x_com = x0 + wp.quat_rotate(r0, body_com[0])

    # linear part
    v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt
    x1 = x_com + v1 * dt

    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(r0, w0)
    tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia * wb)  # coriolis forces

    w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)

    # angular damping
    w1 *= 1.0 - angular_damping * dt

    body_q[state_timestep + 1, robot_idx] = wp.transform(x1 - wp.quat_rotate(r1, body_com[0]), r1)
    body_qd[state_timestep + 1, robot_idx] = wp.spatial_vector(w1, v1)


@wp.kernel
def update_flipper_contacts(flipper_centers: wp.array(dtype=wp.vec3), flipper_angles: wp.array3d(dtype=wp.float32),
                            sim_idx: int, T_s: int, contact_points: wp.array3d(dtype=wp.vec3),
                            flipper_contact_offset: int, dist_increments: float):
    shoot_idx, robot_idx, flipper_idx, contact_idx = wp.tid()

    center = flipper_centers[flipper_idx]
    current_timestep = shoot_idx*T_s + sim_idx
    angle = flipper_angles[current_timestep, robot_idx, flipper_idx]
    dist = dist_increments * wp.float(contact_idx)
    sign = wp.float(1 - 2 * (flipper_idx // 2))
    point = center + wp.vec3(sign*dist * wp.cos(angle), 0.0, -sign*dist * wp.sin(angle))

    target_contact_idx = flipper_contact_offset + contact_idx*4 + flipper_idx
    contact_points[shoot_idx, robot_idx, target_contact_idx] = point

class RenderingState:
    body_q = None

class DiffSim:
    contacts_per_track = 3
    use_flippers = True
    dt = 0.001
    ke = 1.0e3
    kd = 150.0
    kf = 0.5
    renderer = None

    forward_backward_graph = None
    forward_graph = None

    gt_traj = None
    shoot_init_poses = None
    shoot_init_vels = None

    body_q = None
    body_qd = None
    body_f = None

    track_velocities = None
    loss_timesteps = None
    T = None
    T_s = None
    num_shoots = None
    contact_points = None

    rendering_state = None

    def __init__(self, torch_hms, res, use_renderer=False, device="cpu"):
        # instantiate a tracked robot model consisting of a box and collision points
        self.sim_robots = len(torch_hms)  # number of simulated robots is based on number of heightmaps
        self.vis_robots = min(self.sim_robots, 4)

        self.device = device
        self.model, self.contact_points_single, self.flipper_centers, self.flipper_ids = build_track_sim(self.sim_robots, self.vis_robots, self.contacts_per_track, device)

        self.loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)

        self.heightmap_list = self.set_heightmaps(torch_hms, res)
        self.heightmap_array = wp.array(self.heightmap_list, dtype=Heightmap, device=self.device)

        self.heightmap_vis_indices = []
        if use_renderer:
            # instantiate a renderer to render the robot
            opengl_render_settings = dict(scaling=1, near_plane=0.05, far_plane=50.0)
            self.renderer = wp.sim.render.SimRendererOpenGL(
                self.model,
                'WarpSim',
                up_axis="z",
                show_rigid_contact_points=True,
                contact_points_radius=1e-3,
                show_joints=True,
                **opengl_render_settings,
            )

            for robot_idx in range(self.sim_robots):
                current_heigthmap = self.heightmap_list[robot_idx]
                shp = (current_heigthmap.width, current_heigthmap.length)
                self.heightmap_vis_indices.append(get_heightmap_vis_ids(shp))

            # allocate rendering state for n robots and 4 flippers of the first robot
            self.rendering_state = RenderingState()
            self.rendering_state.body_q = wp.zeros((self.vis_robots + 4), dtype=wp.transformf, device=self.device, requires_grad=False)

    def __del__(self):
        if self.renderer is not None:
            self.renderer.clear()

    def set_heightmaps(self, torch_hms, res):
        heightmap_list = []
        for robot_idx in range(self.sim_robots):
            current_hm = torch_hms[robot_idx]
            current_res = res[robot_idx]

            current_shp = current_hm.shape

            current_heightmap = Heightmap()
            current_heightmap.heights = wp.from_torch(current_hm)
            current_heightmap.ke = wp.array(self.ke * np.ones(current_shp), dtype=wp.float32, device=self.device)
            current_heightmap.kd = wp.array(self.kd * np.ones(current_shp), dtype=wp.float32, device=self.device)
            current_heightmap.kf = wp.array(self.kf * np.ones(current_shp), dtype=wp.float32, device=self.device)
            current_heightmap.origin = (-current_shp[0] * current_res / 2, -current_shp[1] * current_res / 2, 0.75)
            current_heightmap.resolution = current_res
            current_heightmap.width = current_shp[0]
            current_heightmap.length = current_shp[1]

            heightmap_list.append(current_heightmap)

        return heightmap_list

    def update_heightmaps(self, torch_hms):
        for robot_idx in range(self.sim_robots):
            current_hm = wp.from_torch(torch_hms[robot_idx])
            self.heightmap_list[robot_idx].heights.assign(current_hm)
        self.heightmap_array.assign(self.heightmap_list)

    def set_T(self, T, T_s=32):
        # T is here the minimum number of timesteps to simulate
        self.T_s = T_s
        self.num_shoots = int(np.ceil(T / T_s))
        self.T = self.num_shoots * T_s

        print('T: ', self.T)
        print('T_s: ', self.T_s)
        print('num_shoots: ', self.num_shoots)

        # init simulation state buffers
        self.body_q = wp.zeros((self.T, self.sim_robots), dtype=wp.transformf, device=self.device, requires_grad=True)
        self.body_qd = wp.zeros((self.T, self.sim_robots), dtype=wp.spatial_vectorf, device=self.device, requires_grad=True)
        self.body_f = wp.zeros((self.T - 1, self.sim_robots), dtype=wp.spatial_vectorf, device=self.device, requires_grad=True)

        self.track_velocities = wp.zeros((self.T, self.sim_robots, 2), dtype=wp.float32, device=self.device, requires_grad=False)
        self.flipper_angles = wp.zeros((self.T, self.sim_robots, 4), dtype=wp.float32, device=self.device, requires_grad=False)

        # copy the single robot contacts into a list of contact points for each shoot
        contact_points_np = np.zeros((self.num_shoots, self.sim_robots, self.contacts_per_track * 8, 3))
        for shoot_idx in range(self.num_shoots):
            contact_points_np[shoot_idx] = self.contact_points_single
        self.contact_points = wp.array(contact_points_np, dtype=wp.vec3, device=self.device, requires_grad=False)

        # fields for debugging of forces and collisions
        constraint_forces = wp.zeros((self.T, self.contacts_per_track * 8), dtype=wp.vec3, device=self.device)
        friction_forces = wp.zeros((self.T, self.contacts_per_track * 8), dtype=wp.vec3, device=self.device)
        contact_positions = wp.zeros((self.T, self.contacts_per_track * 8), dtype=wp.vec3, device=self.device)
        self.contact_info = [constraint_forces, friction_forces, contact_positions]

    def set_target_poses(self, timesteps, poses):
        '''based on a list of stamped poses for each robot, set the simulation ready ground truth trajectories and initial states'''
        assert len(timesteps) == self.sim_robots
        assert len(poses) == self.sim_robots

        self.loss_timesteps = []
        self.gt_trajs = []
        for robot_idx in range(self.sim_robots):
            current_loss_timesteps = timesteps[robot_idx]
            current_gt_traj = poses[robot_idx]
            # convert to warp arrays to allow fast loss computation
            self.loss_timesteps.append(wp.from_numpy(current_loss_timesteps, dtype=wp.int32, device=self.device, requires_grad=False))
            self.gt_trajs.append(wp.from_numpy(current_gt_traj, dtype=wp.transformf, device=self.device, requires_grad=False))

        # TODO: use interpolation instead of nearest neighbour?
        # save initial poses for all simulation shoots
        shoot_init_poses = np.zeros((self.num_shoots, self.sim_robots, 7), dtype=np.float32)
        for robot_idx in range(self.sim_robots):
            for shoot_idx in range(self.num_shoots):
                shoot_start_timestep = shoot_idx * self.T_s
                closest_idx = np.argmin(np.abs(timesteps[robot_idx] - shoot_start_timestep))  # find the stamped pose closest to the shoot start time
                shoot_init_poses[shoot_idx][robot_idx] = poses[robot_idx][closest_idx]  # set initial pose
        self.shoot_init_poses = wp.from_numpy(shoot_init_poses, dtype=wp.transformf, device=self.device)
        self.shoot_init_vels = wp.zeros((self.num_shoots, self.sim_robots), dtype=wp.spatial_vectorf, device=self.device)

    def set_control(self, controls, flipper_angles=None):
        '''given timestamped track controls and initial time of trajectory, parse the controls into simulation ready warp array'''
        self.track_velocities.assign(controls)
        if flipper_angles is not None:
            self.flipper_angles.assign(flipper_angles)
        else:
            self.flipper_angles.zero_()

    def simulate_single(self):
        for field in self.contact_info:
            field.zero_()
        self.body_f.zero_() # zero out forces
        for t in range(self.T - 1):
            self.simulate_flippers_heightmap(t, num_shoots=1)
        self.loss.zero_()  # zero out loss tensor
        self.compute_l2_loss()
        print('single shoot loss: ', self.loss.numpy())
        return self.body_q

    def simulate(self, use_graph=False):
        for field in self.contact_info:
            field.zero_()

        if use_graph:
            if self.device == "cpu":
                raise ValueError("Graph capture is only supported on CUDA devices.")
            if self.forward_graph is None:  # if captured already, run it
                wp.capture_begin()
                try:
                    self.body_f.zero_()  # zero out forces
                    for shoot_t in range(self.T_s - 1):
                        self.simulate_flippers_heightmap(shoot_t)
                finally:
                    self.forward_graph = wp.capture_end()
            wp.capture_launch(self.forward_graph)  # use the existing graph
        else:
            self.body_f.zero_()  # zero out forces
            for shoot_t in range(self.T_s - 1):
                self.simulate_flippers_heightmap(shoot_t)
        return self.body_q

    def continue_velocities(self):
        wp.launch(continue_velocities, dim=(self.num_shoots - 1, self.sim_robots), inputs=[self.body_qd, self.T_s], device=self.device)

    def step_init_vels(self):
        wp.launch(step_init_vels, dim=(self.num_shoots - 1, self.sim_robots), inputs=[self.body_qd, self.T_s, 0.2], device=self.device)

    def init_shoot_states(self):
        wp.launch(init_shoot_poses, dim=(self.num_shoots, self.sim_robots), inputs=[self.body_q, self.shoot_init_poses, self.T_s], device=self.device)
        wp.launch(init_shoot_vels, dim=(self.num_shoots, self.sim_robots), inputs=[self.body_qd, self.shoot_init_vels, self.T_s], device=self.device)

    def save_shoot_init_vels(self):
        wp.launch(save_shoot_init_vels, dim=(self.num_shoots, self.sim_robots), inputs=[self.body_qd, self.shoot_init_vels, self.T_s], device=self.device)

    def simulate_and_backward(self, use_graph=False):
        for field in self.contact_info:
            field.zero_()

        if use_graph:
            if self.device == "cpu":
                raise ValueError("Graph capture is only supported on CUDA devices.")
            if self.forward_backward_graph is None:  # if captured already, run it
                wp.capture_begin()
                try:
                    self.loss.zero_()  # zero out loss tensor
                    self.body_f.zero_()  # zero out forces

                    tape = wp.Tape()  # init tape object to records kernel calls
                    with tape:  # start recording all kernel calls
                        for shoot_t in range(self.T_s - 1):
                            self.simulate_flippers_heightmap(shoot_t)
                        print('compute loss')
                        self.compute_l2_loss()  # given the simulated trajectories, compute the l2 loss
                    print('backward')
                    tape.backward(loss=self.loss)  # propagate gradients into heightmaps
                    tape.zero()  # zero out all variable gradients (except for the heightmaps for some reason...)
                    tape.reset()  # reset tape
                finally:
                    if use_graph and self.forward_backward_graph is None:
                        self.forward_backward_graph = wp.capture_end()
            wp.capture_launch(self.forward_backward_graph)  # use the existing graph

        else:
            self.loss.zero_()  # zero out loss tensor
            self.body_f.zero_()  # zero out forces

            tape = wp.Tape()  # init tape object to records kernel calls
            with tape:  # start recording all kernel calls
                for shoot_t in range(self.T_s - 1):
                    self.simulate_flippers_heightmap(shoot_t)
                print('compute loss')
                self.compute_l2_loss()  # given the simulated trajectories, compute the l2 loss
            print('backward')
            tape.backward(loss=self.loss)  # propagate gradients into heightmaps
            tape.zero()  # zero out all variable gradients (except for the heightmaps for some reason...)
            tape.reset()  # reset tape
        return self.body_q, self.loss

    def simulate_and_backward_torch_tensor(self, torch_hms, use_graph=False):
        torch_hms_warped = wp.from_torch(torch_hms)  # zero copy to warp
        wp.launch(torch_hms_to_warp, dim=torch_hms.size(), inputs=[torch_hms_warped, self.heightmap_array], device=self.device)
        self.simulate_and_backward(use_graph)  # run simulation, loss and backward pass to heightmaps
        for robot_idx in range(self.sim_robots):  # needs to be done in python loop... WTF
            curr_grad = self.heightmap_list[robot_idx].heights.grad
            # and copy the gradients back to the torch tensor
            wp.launch(warp_hm_to_torch, dim=torch_hms_warped.shape[1:], inputs=[curr_grad, torch_hms_warped.grad, robot_idx], device=self.device)
            curr_grad.zero_()  # zero out the gradients after copying
        return self.body_q, self.loss

    def simulate_flippers_heightmap(self, sim_idx, num_shoots=None):
        constraint_forces, friction_forces, contact_points = self.contact_info

        if num_shoots is None:
            num_shoots = self.num_shoots

        if sim_idx % 20 == 0:

            if self.use_flippers:
                flipper_contact_offset = self.contacts_per_track * 4  # where the flipper contacts start in contact_points
                wp.launch(update_flipper_contacts, dim=(num_shoots, self.sim_robots, 4, self.contacts_per_track),
                          inputs=[self.flipper_centers, self.flipper_angles, sim_idx, self.T_s, self.contact_points,
                                  flipper_contact_offset, 0.2/(self.contacts_per_track - 1)], device=self.device)

        # evaluate heightmap collisions for every contact point of each robot
        wp.launch(eval_heightmap_collisions_shoot, dim=(num_shoots, self.sim_robots, self.contacts_per_track * 8),
                  inputs=[self.heightmap_array, self.body_q, self.body_qd, sim_idx, self.T_s,
                          self.track_velocities, self.contact_points, constraint_forces,
                          friction_forces, contact_points, self.body_f], device=self.device)

        wp.launch(
            kernel=integrate_bodies_shoot,
            dim=(num_shoots, self.sim_robots),
            inputs=[
                self.body_q,
                self.body_qd,
                sim_idx,
                self.T_s,
                self.body_f,
                self.model.body_com,
                self.model.body_mass,
                self.model.body_inertia,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
                self.model.gravity,
                0.01,
                self.dt,
            ],
            device=self.device,
        )

    def render_simulation(self, pause=False):
        render_time = 0.0
        body_q_np = self.body_q.numpy()
        if self.renderer is not None:
            for t in range(0, self.T, 50):
                render_time += self.dt
                self.renderer.begin_frame(render_time)
                self.set_visualization_state(body_q_np, t)
                self.renderer.render(self.rendering_state)

                if t == 0:
                    for robot_idx in range(self.vis_robots):
                        current_heigthmap = self.heightmap_list[robot_idx]
                        hm_np = current_heigthmap.heights.numpy()
                        res = current_heigthmap.resolution
                        render_pts = np.array(
                            [np.array(current_heigthmap.origin) + (i * res, j * res, hm_np[i][j])
                             for i in range(current_heigthmap.width) for j in range(current_heigthmap.length)])
                        self.renderer.render_mesh('heightmap%d' % robot_idx, render_pts,
                                                  self.heightmap_vis_indices[robot_idx], smooth_shading=True,
                                                  colors=[0.0, 0.5, 0.0])
                        self.renderer.render_points('heightmap_points%d' % robot_idx, render_pts,
                                                    colors=[[0.2, 0.6, 0.2] for _ in range(len(render_pts))],
                                                    radius=0.02)

                # visualize track contacts
                constraint_forces, friction_forces, contact_positions = self.contact_info

                contact_points_np = contact_positions.numpy()[t]
                self.renderer.render_points('collision_points', contact_points_np,
                                            colors=[[1.0, 1.0, 1.0] for _ in range(len(contact_points_np))],
                                            radius=0.02)

                pts, ids = generate_force_vis(contact_points_np, constraint_forces.numpy()[t])
                self.renderer.render_line_list('constraint_forces', pts, ids, color=(1.0, 0.0, 0.0), radius=0.005)

                pts, ids = generate_force_vis(contact_points_np, friction_forces.numpy()[t], scale=0.01)
                self.renderer.render_line_list('friction_forces', pts, ids, color=(0.0, 0.0, 1.0), radius=0.005)

                self.renderer.end_frame()
            self.renderer.paused = pause

    def compute_l2_loss(self):
        for robot_idx in range(self.sim_robots):
            current_timestamps = self.loss_timesteps[robot_idx]
            currrent_gt_traj = self.gt_trajs[robot_idx]
            wp.launch(lossL2, dim=len(current_timestamps), inputs=[currrent_gt_traj, self.body_q, current_timestamps, robot_idx, self.loss], device=self.device)

    def render_traj(self, xyz, name='gt', color=(0.0, 0.0, 1.0), pause=False):
        if self.renderer is not None:
            self.renderer.render_line_strip(name, xyz, color=color)
            self.renderer.begin_frame(0.0)
            self.renderer.end_frame()
            self.renderer.paused = pause
        else:
            print('No renderer available')

    def render_states(self, name='states', color=(1.0, 0.0, 0.0), pause=False):
        if self.renderer is not None:
            body_q_np = self.body_q.numpy()
            positions = [body_q_np[t, 0, :3] for t in range(self.T)]
            self.renderer.render_line_strip(name, positions, color=color)
            self.renderer.begin_frame(0.0)
            self.renderer.end_frame()
            self.renderer.paused = pause
        else:
            print('No renderer available')

    def render_heightmaps(self, pause=False):
        if self.renderer is not None:
            for robot_idx in range(self.vis_robots):
                current_heigthmap = self.heightmap_list[robot_idx]
                hm_np = current_heigthmap.heights.numpy()
                res = current_heigthmap.resolution
                render_pts = np.array(
                    [np.array(current_heigthmap.origin) + (i * res, j * res, hm_np[i][j])
                     for i in range(current_heigthmap.width) for j in range(current_heigthmap.length)])
                self.renderer.render_mesh('heightmap%d' % robot_idx, render_pts, self.heightmap_vis_indices[robot_idx], smooth_shading=True,
                                          colors=[0.0, 0.5, 0.0])
                self.renderer.render_points('heightmap_points%d' % robot_idx, render_pts,
                                            colors=[[0.2, 0.6, 0.2] for _ in range(len(render_pts))], radius=0.02)

            self.renderer.begin_frame(0.0)
            self.renderer.end_frame()
            self.renderer.paused = pause
        else:
            print('No renderer available')

    def set_visualization_state(self, traj_np, t):
        # simulated state
        flipper_angles = self.flipper_angles.numpy()[t, 0]
        body_q_np = traj_np[t]  # take the current simulation state position
        robot_transform = body_q_np[0]  # transform of the first robot

        # rendering state
        rendering_body_q_np = self.rendering_state.body_q.numpy()
        rendering_body_q_np[:self.vis_robots] = body_q_np[:self.vis_robots]  # copy the current state to the rendering state
        for i in range(4):
            angle = flipper_angles[i]
            pos = np.array(self.flipper_centers.numpy()[i])
            id = self.flipper_ids[i]
            if i >=2:
                angle = angle + np.pi
            quat = R.from_rotvec([0, angle, 0]).as_quat()
            rel_transform = np.concatenate([pos, quat])
            combined = combine_transforms(robot_transform, rel_transform)
            rendering_body_q_np[id] = combined
        self.rendering_state.body_q = wp.from_numpy(rendering_body_q_np, dtype=wp.transformf, device=self.device)


def build_tracked_robots(sim_robots, vis_robots, contacts_per_track, device="cpu"):
    # collision coefficients
    ke = 1.0e5
    kd = 100
    kf = 50
    mu = 0.5

    # dimensions
    center = [0, -0.2]  # center of the track (x, z)
    track_dims = [0.4, 0.5]  # (x, y)
    point_mass = 1 / contacts_per_track  # 1 kg per track divided into the contact points
    track_width = 0.1
    point_radius = 0.02
    contact_point_density = point_mass/(1333 * np.pi * point_radius**3)  # divide by 1000 * 4/3 * pi * r^3
    body_size = [0.5, 0.24, 0.2]  # (x, y, z)
    flipper_len = 0.2

    robot_builder = wp.sim.ModelBuilder(up_vector=(0.0, 0.0, 1.0))
    contact_positions_np = np.zeros((sim_robots, contacts_per_track*(4+4), 3))  # 4 flippers and 2 sides per 2 tracks
    for robot_idx in range(vis_robots):
        main_body = robot_builder.add_body(wp.transform([robot_idx*0.01, robot_idx*0.01, 0.0], [0, 0, 0, 1]), m=35.0, armature=0.01)
        robot_builder.add_shape_box(pos=(0.0, 0.0, 0.0), hx=body_size[0] / 2, hy=body_size[1] / 2, hz=body_size[2] / 2,
                                    body=main_body, ke=ke, kd=kd, kf=kf, mu=mu)
        robot_builder.add_shape_sphere(body=main_body, pos=[body_size[0] / 2 + 0.02, 0, 0], radius=0.02)

        contact_idx = 0
        for i in range(contacts_per_track):
            for side in [-1, 1]:  # left and right rows forming the tracks
                for track in [1, -1]: # left and right tracks
                    pos = [center[0] - track_dims[0]/2 + track_dims[0]/(contacts_per_track-1)*i, track*track_dims[1]/2 + side*track_width/2, center[1]]
                    robot_builder.add_shape_sphere(body=main_body, pos=pos, radius=point_radius, density=contact_point_density, ke=ke, kf=kf, kd=kd, mu=mu)
                    if robot_idx == 0:
                        contact_positions_np[:, contact_idx] = pos
                        contact_idx += 1

    # add 4 flippers for visualization only for the first robot and save body ids
    flipper_ids = []
    flipper_centers = []
    for flipper in [1, -1]:  # front, rear
        for side in [1, -1]:  # left, right
            flipper_center = [center[0] + flipper*(track_dims[0]/2), side*(track_dims[1]/2 + track_width), center[1]]
            flipper_id = robot_builder.add_body(wp.transform(flipper_center, [0, 0, 0, 1]), m=0.1, armature=0.01)
            robot_builder.add_shape_capsule(body=flipper_id, pos=[0.1, 0, 0], radius=0.02, half_height=0.1, up_axis=0)
            flipper_ids.append(flipper_id)
            flipper_centers.append(flipper_center)

    flipper_centers = wp.array(flipper_centers, dtype=wp.vec3, device=device)
    return robot_builder, contact_positions_np, flipper_centers, flipper_ids


def build_track_sim(sim_robots, vis_robots, contacts_per_track, device="cpu"):
    print('Building simulation')
    builder = wp.sim.ModelBuilder(up_vector=(0.0, 0.0, 1.0))
    robot_builder, contact_positions, flipper_centers, flipper_ids = build_tracked_robots(sim_robots, vis_robots, contacts_per_track, device)

    builder.add_builder(robot_builder, wp.transform((0, 0.0, 1.0), wp.quat_identity()))

    print('Finalizing simulation')
    model = builder.finalize(device, requires_grad=True)
    print('Simulation ready')
    return model, contact_positions, flipper_centers, flipper_ids