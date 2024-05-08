import numpy as np
import warp as wp
import warp.sim.render
from scipy.spatial.transform import Rotation as R


def traj_to_line(traj):
    points = []
    for (t, body_q) in traj:
        pos = body_q[:3]
        points.append(pos)
    return points


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
def copy_state(body_q: wp.array2d(dtype=wp.transformf), state_body_q: wp.array(dtype=wp.transformf), sim_idx: int):
    """copy the simulation state body_q into rendering state state_body_q at index sim_idx"""
    tid = wp.tid()
    state_body_q[tid] = body_q[sim_idx][tid]

@wp.kernel
def copy_init_poses(init_poses: wp.array(dtype=wp.transformf), body_q: wp.array2d(dtype=wp.transformf)):
    robot_idx = wp.tid()
    body_q[0, robot_idx] = init_poses[robot_idx]


@wp.kernel
def eval_heightmap_collisions_array(height_map_array: wp.array(dtype=Heightmap),
                                    body_q: wp.array2d(dtype=wp.transformf),
                                    body_qd: wp.array2d(dtype=wp.spatial_vectorf),
                                    sim_idx: int,
                                    track_velocities: wp.array3d(dtype=wp.float32),
                                    contact_points: wp.array2d(dtype=wp.vec3),
                                    constraint_forces: wp.array2d(dtype=wp.vec3),
                                    friction_forces: wp.array2d(dtype=wp.vec3),
                                    collisions: wp.array2d(dtype=wp.vec3),
                                    body_f: wp.array2d(dtype=wp.spatial_vectorf)):
    robot_idx, contact_idx = wp.tid()

    height_map = height_map_array[robot_idx]

    heights = height_map.heights
    kes = height_map.ke
    kds = height_map.kd
    kfs = height_map.kf
    hm_origin = height_map.origin
    hm_res = height_map.resolution
    width = height_map.width
    length = height_map.length

    robot_to_world = body_q[sim_idx, robot_idx]
    robot_to_world_speed = body_qd[sim_idx, robot_idx]
    wheel_to_robot_pos = contact_points[robot_idx, contact_idx]

    forward_to_world = wp.transform_vector(robot_to_world, wp.vec3(1.0, 0.0, 0.0))
    wheel_to_world_pos = wp.transform_point(robot_to_world, wheel_to_robot_pos)
    wheel_to_world_vel = wp.cross(wp.spatial_top(robot_to_world_speed), wheel_to_robot_pos) + wp.spatial_bottom(
        robot_to_world_speed)
    wheel_to_hm = wheel_to_world_pos - hm_origin

    # x, y normalized by the heightmap resolution
    x_n = wheel_to_hm[0] / hm_res
    y_n = wheel_to_hm[1] / hm_res

    u = wp.int(wp.floor(x_n))
    v = wp.int(wp.floor(y_n))

    # hm_height = hm[x_id, z_id]

    if u < 0 or u >= width or v < 0 or v >= length:  # outside heightmap
        return

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
        track_vel = track_velocities[robot_idx, sim_idx, vel_idx]  # left and right track velocities
        tangential_track_velocity = wp.normalize(tangential_track_direction) * track_vel

    # compute the constraint (penetration force) and friction force
    constraint_force = n * (kes[u, v] * d - kds[u, v] * v_n)
    friction_force = -kfs[u, v] * (v_t - tangential_track_velocity) * wp.length(constraint_force)
    total_force = constraint_force + friction_force

    robot_wrench = wp.spatial_vector(wp.cross(wheel_to_robot_pos, total_force), total_force)
    wp.atomic_add(body_f, sim_idx, robot_idx, robot_wrench)

    if robot_idx != 0:
        return

    # Store the contact info only for the first robot TODO: vis all robots?
    constraint_forces[sim_idx, contact_idx] = constraint_force
    friction_forces[sim_idx, contact_idx] = friction_force
    collisions[sim_idx, contact_idx] = wp.vec3(wheel_to_world_pos[0], wheel_to_world_pos[1], hm_height + hm_origin[2])


@wp.kernel
def integrate_bodies_array(
    body_q: wp.array2d(dtype=wp.transform),
    body_qd: wp.array2d(dtype=wp.spatial_vector),
    sim_idx: int,
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
    tid = wp.tid()

    # positions
    q = body_q[sim_idx, tid]
    qd = body_qd[sim_idx, tid]
    f = body_f[sim_idx, tid]

    # masses
    mass = m[tid]
    inv_mass = inv_m[tid]  # 1 / mass

    inertia = I[tid]
    inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix

    # unpack transform
    x0 = wp.transform_get_translation(q)
    r0 = wp.transform_get_rotation(q)

    # unpack spatial twist
    w0 = wp.spatial_top(qd)
    v0 = wp.spatial_bottom(qd)

    # unpack spatial wrench
    t0 = wp.spatial_top(f)
    f0 = wp.spatial_bottom(f)

    x_com = x0 + wp.quat_rotate(r0, body_com[tid])

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

    body_q[sim_idx + 1, tid] = wp.transform(x1 - wp.quat_rotate(r1, body_com[tid]), r1)
    body_qd[sim_idx + 1, tid] = wp.spatial_vector(w1, v1)

@wp.kernel
def update_flipper_contacts(flipper_centers: wp.array(dtype=wp.vec3), flipper_angles: wp.array3d(dtype=wp.float32), sim_idx: int,
                            contact_points: wp.array2d(dtype=wp.vec3), flipper_contact_offset: int, dist_increments: float):
    robot_idx, flipper_idx, contact_idx = wp.tid()

    center = flipper_centers[flipper_idx]
    angle = flipper_angles[robot_idx, sim_idx, flipper_idx]
    dist = dist_increments * wp.float(contact_idx)
    sign = wp.float(1 - 2 * (flipper_idx // 2))
    point = center + wp.vec3(sign*dist * wp.cos(angle), 0.0, -sign*dist * wp.sin(angle))

    target_idx = flipper_contact_offset + contact_idx*4 + flipper_idx
    contact_points[robot_idx, target_idx] = point

class RenderingState:
    body_q = None

class TrackSimulator:
    contacts_per_track = 3
    use_flippers = True
    dt = 0.001
    renderer = None

    cuda_graph = None

    body_q = None
    body_qd = None
    body_f = None

    track_velocities = None
    rendering_state = None

    def __init__(self, np_hms, np_kfs, res, T=10, use_renderer=False, device="cpu"):
        # instantiate a tracked robot model consisting of a box and collision points
        self.n_robots = len(np_hms)  # number of simulated robots is based on number of heightmaps
        self.device = device
        self.model, self.contact_points, self.flipper_centers, self.flipper_ids = build_track_sim(self.n_robots, self.contacts_per_track, device)

        # init fields for simulation
        self.T = T
        self.body_q = wp.zeros((T + 1, self.n_robots), dtype=wp.transformf, device=self.device, requires_grad=False)
        self.body_qd = wp.zeros((T + 1, self.n_robots), dtype=wp.spatial_vectorf, device=self.device, requires_grad=False)
        self.body_f = wp.zeros((T, self.n_robots), dtype=wp.spatial_vectorf, device=self.device, requires_grad=False)

        self.heightmap_list = []
        for robot_idx in range(self.n_robots):
            current_hm = np_hms[robot_idx]
            current_kf = np_kfs[robot_idx]
            current_res = res[robot_idx]

            current_shp = current_hm.shape

            current_heightmap = Heightmap()
            current_heightmap.heights = wp.array(current_hm, dtype=wp.float32, device=self.device)
            current_heightmap.ke = wp.array(1.0e4 * np.ones(current_shp), dtype=wp.float32, device=self.device)
            current_heightmap.kd = wp.array(150.0 * np.ones(current_shp), dtype=wp.float32, device=self.device)
            current_heightmap.kf = wp.array(current_kf, dtype=wp.float32, device=self.device)
            current_heightmap.origin = (-current_shp[0] * current_res / 2, -current_shp[1] * current_res / 2, 0.0)
            current_heightmap.resolution = current_res
            current_heightmap.width = current_shp[0]
            current_heightmap.length = current_shp[1]

            self.heightmap_list.append(current_heightmap)
        self.heightmap_array = wp.array(self.heightmap_list, dtype=Heightmap, device=self.device)

        self.heightmap_vis_indices = []
        if use_renderer:
            # instantiate a renderer to render the robot
            opengl_render_settings = dict(scaling=1, near_plane=0.01)
            self.renderer = wp.sim.render.SimRendererOpenGL(
                self.model,
                'WarpSim',
                up_axis="z",
                show_rigid_contact_points=True,
                contact_points_radius=1e-3,
                show_joints=True,
                **opengl_render_settings,
            )

            for robot_idx in range(self.n_robots):
                current_heigthmap = self.heightmap_list[robot_idx]
                shp = (current_heigthmap.width, current_heigthmap.length)
                self.heightmap_vis_indices.append(get_heightmap_vis_ids(shp))

            # allocate rendering state for n robots and 4 flippers of the first robot
            self.rendering_state = RenderingState()
            self.rendering_state.body_q = wp.zeros((self.n_robots + 4,), dtype=wp.transformf, device=self.device, requires_grad=False)

        # fields for debugging of forces and collisions
        constraint_forces = wp.zeros((T, self.contacts_per_track*8), dtype=wp.vec3, device=self.device, requires_grad=False)
        friction_forces = wp.zeros((T, self.contacts_per_track*8), dtype=wp.vec3, device=self.device, requires_grad=False)
        contact_positions = wp.zeros((T, self.contacts_per_track*8), dtype=wp.vec3, device=self.device, requires_grad=False)
        self.contact_info = [constraint_forces, friction_forces, contact_positions]

    def __del__(self):
        if self.renderer is not None:
            self.renderer.clear()

    def set_control(self, control_np, flipper_angles_np):
        assert control_np.shape == (self.n_robots, self.T, 2)
        assert flipper_angles_np.shape == (self.n_robots, self.T, 4)
        self.track_velocities = wp.array(control_np, dtype=wp.float32, device=self.device, requires_grad=False)
        self.flipper_angles = wp.array(flipper_angles_np, dtype=wp.float32, device=self.device, requires_grad=False)

    def set_init_poses(self, init_poses):
        assert init_poses.shape == (self.n_robots, 7)
        wp_poses = wp.array(init_poses, dtype=wp.transformf, device=self.device, requires_grad=False)
        wp.launch(copy_init_poses, dim=self.n_robots, inputs=[wp_poses, self.body_q], device=self.device)

    def simulate(self, render=False, use_graph=False):

        assert self.body_q is not None
        assert self.body_qd is not None
        assert self.body_f is not None
        assert self.track_velocities is not None
        assert self.flipper_angles is not None

        if render:
            render_time = 0.0

        if use_graph:
            if self.device == "cpu":
                raise ValueError("Graph capture is only supported on CUDA devices.")
            if self.cuda_graph is None:  # construct the cuda graph
                wp.capture_begin()
                try:
                    self.body_f.zero_()  # zero out forces
                    for field in self.contact_info:  # zero out contact info for visualization
                        field.zero_()
                    for t in range(self.T):
                        self.simulate_flippers_heightmap(t)
                finally:
                    self.cuda_graph = wp.capture_end()
            wp.capture_launch(self.cuda_graph)  # use the existing graph
        else:
            self.body_f.zero_()  # zero out forces
            for t in range(self.T):
                self.simulate_flippers_heightmap(t)

        if render and self.renderer is not None:
            constraint_forces, friction_forces, contact_positions = self.contact_info
            constraint_forces = constraint_forces.numpy()
            friction_forces = friction_forces.numpy()
            contact_positions = contact_positions.numpy()

            for t in range(self.T):
                render_time += self.dt
                self.renderer.begin_frame(render_time)
                self.set_visualization_state(t)  # set flipper states for rendering
                self.renderer.render(self.rendering_state)

                if t == 0:
                    for robot_idx in range(self.n_robots):
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
                                                    colors=[[0.2, 0.6, 0.2] for _ in range(len(render_pts))], radius=0.02)

                # visualize track contacts
                contact_points_t = contact_positions[t]
                self.renderer.render_points('collision_points', contact_points_t,
                                            colors=[[1.0, 1.0, 1.0] for _ in range(len(contact_points_t))], radius=0.02)

                pts, ids = generate_force_vis(contact_points_t, constraint_forces[t])
                self.renderer.render_line_list('constraint_forces', pts, ids, color=(1.0, 0.0, 0.0), radius=0.005)

                pts, ids = generate_force_vis(contact_points_t, friction_forces[t], scale=0.01)
                self.renderer.render_line_list('friction_forces', pts, ids, color=(0.0, 0.0, 1.0), radius=0.005)

                self.renderer.end_frame()
                self.renderer.paused = False

        return self.body_q

    def simulate_flippers_heightmap(self, sim_idx):
        constraint_forces, friction_forces, contact_points = self.contact_info

        if self.use_flippers:  # update collision points with flipper_angles
            flipper_contact_offset = self.contacts_per_track * 4
            wp.launch(update_flipper_contacts, dim=(self.n_robots, 4, self.contacts_per_track),
                      inputs=[self.flipper_centers, self.flipper_angles, sim_idx, self.contact_points,
                              flipper_contact_offset, 0.2/(self.contacts_per_track - 1)], device=self.device)

            # evaluate heightmap collisions for every contact point of each robot
            wp.launch(eval_heightmap_collisions_array, dim=self.contact_points.shape,
                      inputs=[self.heightmap_array, self.body_q, self.body_qd, sim_idx, self.track_velocities,
                              self.contact_points, constraint_forces, friction_forces, contact_points, self.body_f],
                      device=self.device)

            wp.launch(
                kernel=integrate_bodies_array,
                dim=self.n_robots,
                inputs=[
                    self.body_q,
                    self.body_qd,
                    sim_idx,
                    self.body_f,
                    self.model.body_com,
                    self.model.body_mass,
                    self.model.body_inertia,
                    self.model.body_inv_mass,
                    self.model.body_inv_inertia,
                    self.model.gravity,
                    0.05,
                    self.dt,
                ],
                device=self.device,
            )

    def render_traj(self, trajectory, name='gt', color=(0.0, 0.0, 1.0), pause=False):
        if self.renderer is not None:
            self.renderer.render_line_strip(name, traj_to_line(trajectory), color=color)
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
            for robot_idx in range(self.n_robots):
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

    def set_visualization_state(self, sim_idx):
        # simulated state
        flipper_angles = self.flipper_angles.numpy()[0][sim_idx]
        body_q_np = self.body_q.numpy()[sim_idx]  # take the current simulation state position
        robot_transform = body_q_np[0]  # transform of the first robot

        # rendering state
        rendering_body_q_np = self.rendering_state.body_q.numpy()
        rendering_body_q_np[:self.n_robots] = body_q_np  # copy the current state to the rendering state
        for i in range(4):
            angle = flipper_angles[i]
            pos = np.array(self.flipper_centers.numpy()[i])
            id = self.flipper_ids[i]
            if i >= 2:
                angle = angle + np.pi
            quat = R.from_rotvec([0, angle, 0]).as_quat()
            rel_transform = np.concatenate([pos, quat])
            combined = combine_transforms(robot_transform, rel_transform)
            rendering_body_q_np[id] = combined
        self.rendering_state.body_q = wp.from_numpy(rendering_body_q_np, dtype=wp.transformf, device=self.device)


def build_tracked_robots(n_robots, contacts_per_track, device="cpu"):
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
    contact_positions_np = np.zeros((n_robots, contacts_per_track*(4+4), 3))  # 4 flippers and 2 sides per 2 tracks
    for robot_idx in range(n_robots):
        main_body = robot_builder.add_body(wp.transform([robot_idx*0.01, robot_idx*0.01, 0.0], [0, 0, 0, 1]), m=35.0, armature=0.01)
        robot_builder.add_shape_box(pos=(0.0, 0.0, 0.0), hx=body_size[0] / 2, hy=body_size[1] / 2, hz=body_size[2] / 2,
                                    body=main_body, ke=ke, kd=kd, kf=kf, mu=mu)
        robot_builder.add_shape_sphere(body=main_body, pos=[body_size[0] / 2 + 0.02, 0, 0], radius=0.02)  # mark the front of the robot

        # add contact points for each robot
        contact_idx = 0
        for i in range(contacts_per_track):
            for side in [-1, 1]:  # left and right rows forming the tracks
                for track in [1, -1]:  # left and right tracks
                    pos = [center[0] - track_dims[0] / 2 + track_dims[0] / (contacts_per_track - 1) * i,
                           track * track_dims[1] / 2 + side * track_width / 2, center[1]]
                    robot_builder.add_shape_sphere(body=main_body, pos=pos, radius=point_radius,
                                                   density=contact_point_density, ke=ke, kf=kf, kd=kd, mu=mu)
                    contact_positions_np[robot_idx, contact_idx] = pos
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
    contact_positions = wp.array(contact_positions_np, dtype=wp.vec3, device=device)
    return robot_builder, contact_positions, flipper_centers, flipper_ids


def build_track_sim(n_robots, contacts_per_track, device="cpu"):
    print('Building simulation')
    builder = wp.sim.ModelBuilder(up_vector=(0.0, 0.0, 1.0))
    robot_builder, contact_positions, flipper_centers, flipper_ids = build_tracked_robots(n_robots, contacts_per_track, device)

    builder.add_builder(robot_builder, wp.transform((0, 0.0, 1.0), wp.quat_identity()))
    builder.num_rigid_contacts_per_env = 0

    print('Finalizing simulation')
    model = builder.finalize(device, requires_grad=False)
    print('Simulation ready')
    return model, contact_positions, flipper_centers, flipper_ids