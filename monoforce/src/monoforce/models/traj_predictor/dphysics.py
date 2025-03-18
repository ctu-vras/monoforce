import numpy as np
import torch
from torchdiffeq import odeint
from .dphys_config import DPhysConfig


def normalized(x, eps=1e-6, dim=-1):
    """
    Normalizes the input tensor.

    Parameters:
    - x: Input tensor.
    - eps: Small value to avoid division by zero.

    Returns:
    - Normalized tensor.
    """
    norm = torch.norm(x, dim=dim, keepdim=True)
    return x / torch.clamp(norm, min=eps)


def skew_symmetric(v):
    """
    Returns the skew-symmetric matrix of a vector.

    Parameters:
    - v: Input vector.

    Returns:
    - Skew-symmetric matrix of the input vector.
    """
    assert v.dim() == 2 and v.shape[1] == 3
    U = torch.zeros(v.shape[0], 3, 3, device=v.device)
    U[:, 0, 1] = -v[:, 2]
    U[:, 0, 2] = v[:, 1]
    U[:, 1, 2] = -v[:, 0]
    U[:, 1, 0] = v[:, 2]
    U[:, 2, 0] = -v[:, 1]
    U[:, 2, 1] = v[:, 0]
    return U

def generate_controls(n_trajs=10,
                      time_horizon=5.0, dt=0.01,
                      v_range=(-1.0, 1.0), w_range=(-1.0, 1.0)):
    """
    Generates control inputs for the robot trajectories.

    Parameters:
    - n_trajs: Number of trajectories.
    - time_horizon: Time horizon for each trajectory.
    - dt: Time step.
    - v_range: Range of the forward speed.
    - w_range: Range of the rotational speed.

    Returns:
    - Linear and angular velocities for the robot trajectories: (n_trajs, time_steps, 2).
    - Time stamps for the trajectories.
    """
    N = int(time_horizon / dt)
    time_stamps = torch.linspace(0, time_horizon, N)

    v = torch.rand(n_trajs) * (v_range[1] - v_range[0]) + v_range[0]  # Forward speed
    w = torch.rand(n_trajs) * (w_range[1] - w_range[0]) + w_range[0]  # Rotational speed

    # repeat the control inputs for each time step
    v = v.unsqueeze(1).repeat(1, N)
    w = w.unsqueeze(1).repeat(1, N)

    # stack the control inputs
    control_inputs = torch.stack([v, w], dim=-1)

    return control_inputs, time_stamps


def vw_to_track_vels(v, w, robot_size, n_tracks):
    """
    Converts the forward and rotational speeds to track velocities.

    Parameters:
    - v: Forward speed.
    - w: Rotational speed.
    - robot_size: Size of the robot.
    - n_tracks: Number of tracks (2 or 4).

    Returns:
    - Track velocities.
    """
    Lx, Ly = robot_size
    if n_tracks == 2:
        v_L = v - w * (Ly / 2.0)  # Left wheel velocity
        v_R = v + w * (Ly / 2.0)  # Right wheel velocity
        # left, right
        track_vels = torch.stack([v_L, v_R], dim=-1)
    elif n_tracks == 4:
        # front left, front right, rear left, rear right
        v_FL = v - w * Ly / 2.0
        v_FR = v + w * Ly / 2.0
        v_RL = v - w * Ly / 2.0
        v_RR = v + w * Ly / 2.0
        track_vels = torch.stack([v_FL, v_FR, v_RL, v_RR], dim=-1)
    else:
        raise ValueError('n_tracks must be 2 or 4')

    return track_vels


def inertia_tensor(mass, points):
    """
    Compute the inertia tensor for a rigid body represented by point masses.

    Parameters:
    mass (float): The total mass of the body.
    points (tensor): A tensor of points (x, y, z) representing the mass distribution.
                     Each point contributes equally to the total mass, BxNx3.

    Returns:
    torch.Tensor: A Bx3x3 inertia tensor matrix.
    """
    # Number of points
    assert points.dim() == 3
    n_points = points.shape[1]
    B = points.shape[0]

    # Mass per point: assume uniform mass distribution
    mass_per_point = mass / n_points

    # Compute the inertia tensor components
    Ixx = torch.sum(mass_per_point * (points[:, :, 1] ** 2 + points[:, :, 2] ** 2), dim=1)
    Iyy = torch.sum(mass_per_point * (points[:, :, 0] ** 2 + points[:, :, 2] ** 2), dim=1)
    Izz = torch.sum(mass_per_point * (points[:, :, 0] ** 2 + points[:, :, 1] ** 2), dim=1)
    Ixy = -torch.sum(mass_per_point * points[:, :, 0] * points[:, :, 1], dim=1)
    Ixz = -torch.sum(mass_per_point * points[:, :, 0] * points[:, :, 2], dim=1)
    Iyz = -torch.sum(mass_per_point * points[:, :, 1] * points[:, :, 2], dim=1)

    # Construct the inertia tensor matrix
    I = torch.stack([torch.stack([Ixx, Ixy, Ixz], dim=1),
                     torch.stack([Ixy, Iyy, Iyz], dim=1),
                     torch.stack([Ixz, Iyz, Izz], dim=1)], dim=1)
    assert I.shape == (B, 3, 3)

    return I


class DPhysics(torch.nn.Module):
    def __init__(self, dphys_cfg=DPhysConfig(), device='cpu'):
        super(DPhysics, self).__init__()
        self.dphys_cfg = dphys_cfg
        self.device = device
        self.x_points = self.dphys_cfg.robot_points.to(self.device).unsqueeze(0)  # robot body points, (1, N, 3)

        # 1x3x3 inertia tensor, kg*m^2
        self.I = inertia_tensor(mass=self.dphys_cfg.robot_mass, points=self.x_points).to(self.device)
        self.I_inv = torch.linalg.inv(self.I)  # inverse of the inertia tensor

        # terrain properties: heightmap, stiffness, damping, friction
        self.z_grid = None
        self.friction = None
        self.stiffness = dphys_cfg.stiffness
        self.damping = dphys_cfg.damping

        # control inputs and joint angles
        self.controls = None
        self.joint_angles = None

        # simulation (prediction) parameters: time horizon and step size
        T, dt = self.dphys_cfg.traj_sim_time, self.dphys_cfg.dt
        self.ts = torch.linspace(0, T, int(T / dt)).to(self.device)

        # integration method: odeint or custom
        self.integrator = self.dynamics_odeint if self.dphys_cfg.use_odeint else self.dynamics

    def forward_kinematics(self, t, state):
        # unpack state
        x, xd, R, omega = state
        B = x.shape[0]
        N_pts = self.x_points.shape[1]
        assert x.shape == (B, 3)
        assert xd.shape == (B, 3)
        assert R.shape == (B, 3, 3)
        assert omega.shape == (B, 3)

        # closest time step in the control inputs
        t_id = torch.argmin(torch.abs(t - self.ts))
        controls_t = self.controls[:, t_id]
        assert controls_t.shape == (B, 2)
        assert controls_t.shape[1] == 2  # linear and angular velocities
        joint_angles_t = self.joint_angles[:, t_id]
        assert joint_angles_t.shape == (B, 4)
        assert self.z_grid.dim() == 3  # (B, H, W)

        # update the robot body points based on the joint angles
        x_points = self.update_joints(joint_angles_t)
        assert x_points.shape == (B, N_pts, 3)

        # update the inertia tensor based on the new robot body points configuration
        I = inertia_tensor(mass=self.dphys_cfg.robot_mass, points=x_points)
        self.I_inv = torch.linalg.inv(I)  # inverse of the inertia tensor

        # motion of point composed of cog motion and rotation of the rigid body
        x_points = x_points @ R.transpose(1, 2) + x.unsqueeze(1)

        # motion of point composed of cog motion and rotation of the rigid body
        # Koenig's theorem in mechanics: v_i = v_cog + omega x (r_i - r_cog)
        xd_points = xd.unsqueeze(1) + torch.linalg.cross(omega.unsqueeze(1), x_points - x.unsqueeze(1))
        assert xd_points.shape == (B, N_pts, 3)

        for p in self.dphys_cfg.driving_parts:
            assert p.dim() == 1 and p.shape[0] == x_points.shape[1]  # (N,)

        # compute the terrain properties at the robot points
        z_points, n = self.interpolate_grid(self.z_grid, x_points[..., 0], x_points[..., 1], return_normals=True)
        z_points = z_points.unsqueeze(-1)
        assert z_points.shape == (B, N_pts, 1)
        assert n.shape == (B, N_pts, 3)

        friction_ceofs = self.interpolate_grid(self.friction, x_points[..., 0], x_points[..., 1]).unsqueeze(-1)
        assert friction_ceofs.shape == (B, N_pts, 1)

        # check if the rigid body is in contact with the terrain
        dh_points = x_points[..., 2:3] - z_points
        # in_contact = dh_points < 0.
        # soft contact model
        in_contact = torch.sigmoid(-10. * dh_points)
        assert in_contact.shape == (B, N_pts, 1)

        # reaction at the contact points as spring-damper forces
        m, g = self.dphys_cfg.robot_mass, self.dphys_cfg.gravity
        xd_points_n = (xd_points * n).sum(dim=2).unsqueeze(2)  # normal velocity
        assert xd_points_n.shape == (B, N_pts, 1)
        F_reaction = -torch.mul((self.stiffness * dh_points + self.damping * xd_points_n), n)  # F_s = -k * dh - b * v_n
        n_contact_pts = torch.sum(in_contact, dim=1, keepdim=True)
        F_reaction = torch.mul(F_reaction, in_contact) / n_contact_pts  # apply forces only at the contact points
        F_reaction = torch.clamp(F_reaction, min=-m*g, max=m*g)
        assert F_reaction.shape == (B, N_pts, 3)

        # static and dynamic friction forces: https://en.wikipedia.org/wiki/Friction
        thrust_dir = normalized(R[..., 0])  # direction of the thrust
        N = torch.norm(F_reaction, dim=2)  # normal force magnitude at the contact points
        track_vels = vw_to_track_vels(v=controls_t[:, 0], w=controls_t[:, 1],
                                      robot_size=self.dphys_cfg.robot_size, n_tracks=len(self.dphys_cfg.driving_parts))
        assert track_vels.shape == (B, len(self.dphys_cfg.driving_parts))
        cmd_vels = torch.zeros_like(xd_points)
        for i in range(len(self.dphys_cfg.driving_parts)):
            mask = self.dphys_cfg.driving_parts[i]
            u = track_vels[:, i].unsqueeze(1) * thrust_dir
            cmd_vels[:, mask] = u.unsqueeze(1)
        slip_vel = friction_ceofs * (cmd_vels - xd_points)
        slip_vel_n = (slip_vel * n).sum(dim=2).unsqueeze(2)  # normal velocity difference
        slip_vel_tau = slip_vel - slip_vel_n * n  # tangential velocity difference
        F_friction = N.unsqueeze(2) * slip_vel_tau  # F_f = mu * N * v_slip
        F_friction = torch.clamp(F_friction, min=-m*g, max=m*g)
        assert F_friction.shape == (B, N_pts, 3)

        # rigid body rotation: M = sum(r_i x F_i)
        torque = torch.sum(torch.linalg.cross(x_points - x.unsqueeze(1), F_reaction + F_friction), dim=1)
        omega_d = (self.I_inv @ torque.unsqueeze(2)).squeeze(2)  # omega_d = I^(-1) M
        omega_d = torch.clamp(omega_d, min=-self.dphys_cfg.omega_max, max=self.dphys_cfg.omega_max)
        Omega_skew = skew_symmetric(omega)  # Omega_skew = [omega]_x
        dR = Omega_skew @ R  # dR = [omega]_x R
        assert omega_d.shape == (B, 3)
        assert dR.shape == (B, 3, 3)

        # motion of the cog
        F_grav = m * g * torch.as_tensor(self.dphys_cfg.gravity_direction, device=self.device).unsqueeze(0)  # F_grav = [0, 0, -m * g]
        F_cog = F_grav + F_reaction.sum(dim=1) + F_friction.sum(dim=1)  # ma = sum(F_i)
        xdd = F_cog / m  # a = F / m
        assert xdd.shape == (B, 3)

        dstate = (xd, xdd, dR, omega_d)
        forces = (F_reaction, F_friction)

        return dstate, forces

    def update_state(self, state, dstate, dt):
        """
        Integrates the states of the rigid body for the next time step.
        """
        x, xd, R, omega = state
        _, xdd, dR, omega_d = dstate

        xd = self.integration_step(xd, xdd, dt, mode=self.dphys_cfg.integration_mode)
        x = self.integration_step(x, xd, dt, mode=self.dphys_cfg.integration_mode)
        omega = self.integration_step(omega, omega_d, dt, mode=self.dphys_cfg.integration_mode)
        R = self.integrate_rotation(R, omega, dt)

        state = (x, xd, R, omega)

        return state

    @staticmethod
    def integrate_rotation(R, omega, dt, eps=1e-6):
        """
        Integrates the rotation matrix for the next time step using Rodrigues' formula.

        Parameters:
        - R: Tensor of rotation matrices.
        - omega: Tensor of angular velocities.
        - dt: Time step.
        - eps: Small value to avoid division by zero.

        Returns:
        - Updated rotation matrices.

        Reference:
            https://math.stackexchange.com/questions/167880/calculating-new-rotation-matrix-with-its-derivative-given
        """
        assert R.dim() == 3 and R.shape[-2:] == (3, 3)
        assert omega.dim() == 2 and omega.shape[1] == 3
        assert dt > 0

        # Compute the skew-symmetric matrix of the angular velocities
        Omega_x = skew_symmetric(omega)

        # Compute exponential map of the skew-symmetric matrix
        theta = torch.norm(omega, dim=-1, keepdim=True).unsqueeze(-1)

        # Normalize the angular velocities
        Omega_x_norm = Omega_x / torch.clamp(theta, min=eps)

        # Rodrigues' formula: R_new = R * (I + Omega_x * sin(theta * dt) + Omega_x^2 * (1 - cos(theta * dt)))
        I = torch.eye(3).to(R.device)
        R_new = R @ (I + Omega_x_norm * torch.sin(theta * dt) + Omega_x_norm @ Omega_x_norm * (1 - torch.cos(theta * dt)))

        return R_new

    def update_joints(self, joint_angles):
        """
        Rotate the driving parts according to the joint angles

        Parameters:
        - joint_angles: Joint angles, [fl, fr, rr, rl].
        - joint_positions: Joint positions, [xyz_fl, xyz_fr, xyz_rl, xyz_rr], xyz.shape = (3,).
        - x_points: Robot body points, (N, 3).
        - driving_parts: List of driving parts, [[fl], [fr], [rr], [rl]].
        """
        B = joint_angles.shape[0]
        x_points = self.x_points.repeat(B, 1, 1)  # (B, N, 3)

        # TODO: add support for other robots, not only marv
        if self.dphys_cfg.robot != 'marv' or torch.allclose(joint_angles, torch.zeros_like(joint_angles)):
            return x_points

        driving_parts = self.dphys_cfg.driving_parts
        joint_positions = list(self.dphys_cfg.joint_positions.values())
        for i in range(len(driving_parts)):
            # rotate around y-axis of the joint position
            xyz = torch.as_tensor(joint_positions[i], dtype=x_points.dtype, device=self.device).unsqueeze(0)
            angle = joint_angles[:, i]
            R = torch.stack([torch.cos(angle), torch.zeros_like(angle), torch.sin(angle),
                            torch.zeros_like(angle), torch.ones_like(angle), torch.zeros_like(angle),
                            -torch.sin(angle), torch.zeros_like(angle), torch.cos(angle)], dim=1).view(B, 3, 3)
            mask = driving_parts[i]
            points = x_points[:, mask]
            points = points - xyz.unsqueeze(1)
            points = points @ R.transpose(1, 2)
            points = points + xyz.unsqueeze(1)
            x_points[:, mask] = points
        return x_points

    @staticmethod
    def integration_step(x, xd, dt, mode='euler'):
        """
        Performs an integration step using the Euler method.

        Parameters:
        - x: Tensor of positions.
        - xd: Tensor of velocities.
        - dt: Time step.

        Returns:
        - Updated positions and velocities.
        """
        if mode == 'euler':
            x = x + xd * dt
        elif mode == 'rk4':
            k1 = dt * xd
            k2 = dt * (xd + k1 / 2)
            k3 = dt * (xd + k2 / 2)
            k4 = dt * (xd + k3)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            raise ValueError(f'Unknown integration mode: {mode}')
        return x

    def interpolate_grid(self, grid, x_query, y_query, return_normals=False):
        """
        Interpolates the height at the desired (x_query, y_query) coordinates.

        Parameters:
        - grid: Tensor of grid values corresponding to the x and y coordinates (3D array), (B, H, W).
        - x_query: Tensor of desired x coordinates for interpolation (2D array), (B, N).
        - y_query: Tensor of desired y coordinates for interpolation (2D array), (B, N).
        - return_normals: Bool to return the surface normals at the queried coordinates or not.

        Returns:
        - Interpolated grid values at the queried coordinates, (B, N).
        - Surface normals at the queried coordinates (if return_normals=True), (B, N, 3).
        """
        # unpack config
        d_max = self.dphys_cfg.d_max
        grid_res = self.dphys_cfg.grid_res

        # Ensure inputs are tensors
        grid = torch.as_tensor(grid)
        x_query = torch.as_tensor(x_query)
        y_query = torch.as_tensor(y_query)

        # Get the grid dimensions
        B, H, W = grid.shape

        # Flatten the grid coordinates
        z_grid_flat = grid.reshape(B, -1)

        # Flatten the query coordinates
        x_query_flat = x_query.reshape(B, -1)
        y_query_flat = y_query.reshape(B, -1)

        # Compute the indices of the grid points surrounding the query points
        x_i = ((x_query_flat + d_max) / grid_res).long()
        y_i = ((y_query_flat + d_max) / grid_res).long()

        # Compute the fractional part of the indices
        x_frac = (x_query_flat + d_max) / grid_res - x_i.float()
        y_frac = (y_query_flat + d_max) / grid_res - y_i.float()

        # Compute the indices of the grid points
        i_c = y_i + H * x_i
        i_f = y_i + H * (x_i + 1)
        i_l = (y_i + 1) + H * x_i
        i_fl = (y_i + 1) + H * (x_i + 1)
        # Clamp the indices to avoid out-of-bound errors
        i_c = torch.clamp(i_c, 0, H * W - 1)
        i_f = torch.clamp(i_f, 0, H * W - 1)
        i_l = torch.clamp(i_l, 0, H * W - 1)
        i_fl = torch.clamp(i_fl, 0, H * W - 1)

        # Interpolate the z values (linear interpolation)
        z_center = z_grid_flat.gather(1, i_c)
        z_front = z_grid_flat.gather(1, i_f)
        z_left = z_grid_flat.gather(1, i_l)
        z_front_left = z_grid_flat.gather(1, i_fl)
        z_query = (1 - x_frac) * (1 - y_frac) * z_center + \
                  (1 - x_frac) * y_frac * z_front + \
                  x_frac * (1 - y_frac) * z_left + \
                  x_frac * y_frac * z_front_left

        if return_normals:
            # Estimate normals using the height map
            dz_dx = (z_front - z_center) / grid_res
            dz_dy = (z_left - z_center) / grid_res
            n = torch.stack([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=-1)
            n = normalized(n)
            return z_query, n

        return z_query

    def forward_kinematics_extended_state(self, t, state_extended):
        """
        Extended forward kinematics function that takes the extended state as input.
        """
        x, xd, R, omega, F_spring, F_friction = state_extended
        state = (x, xd, R, omega)
        dstate, forces = self.forward_kinematics(t, state)
        dstate = dstate + forces
        return dstate

    def dynamics(self, state):
        Xs, Xds, Rs, Omegas, F_springs, F_frictions = [], [], [], [], [], []
        for t in self.ts:
            # forward kinematics
            dstate, forces = self.forward_kinematics(t=t, state=state)
            # update state: integration steps
            state = self.update_state(state, dstate, self.dphys_cfg.dt)

            # unpack state, its differential, and forces
            x, xd, R, omega = state
            F_spring, F_friction = forces

            # save states
            Xs.append(x)
            Xds.append(xd)
            Rs.append(R)
            Omegas.append(omega)

            # save forces
            F_springs.append(F_spring)
            F_frictions.append(F_friction)

        # stack the states and forces
        Xs = torch.stack(Xs, dim=1)
        Xds = torch.stack(Xds, dim=1)
        Rs = torch.stack(Rs, dim=1)
        Omegas = torch.stack(Omegas, dim=1)
        F_springs = torch.stack(F_springs, dim=1)
        F_frictions = torch.stack(F_frictions, dim=1)

        return Xs, Xds, Rs, Omegas, F_springs, F_frictions

    def dynamics_odeint(self, state):
        """
        Simulates the dynamics of the robot using ODE solver.
        """
        B = state[0].shape[0]
        N_pts = self.x_points.shape[1]
        N_ts = len(self.ts)
        f_spring = torch.zeros(B, N_pts, 3, device=self.device)
        f_friction = torch.zeros(B, N_pts, 3, device=self.device)
        forces = (f_spring, f_friction)
        state_extended = tuple(state) + tuple(forces)
        state_extended = odeint(self.forward_kinematics_extended_state, state_extended, self.ts,
                                method=self.dphys_cfg.integration_mode, rtol=1e-3, atol=1e-3)
        Xs, Xds, Rs, Omegas, F_springs, F_frictions = state_extended

        # transpose the states and forces to (B, N, D)
        Xs = Xs.permute(1, 0, 2)
        assert Xs.shape == (B, N_ts, 3)
        Xds = Xds.permute(1, 0, 2)
        assert Xds.shape == (B, N_ts, 3)
        Rs = Rs.permute(1, 0, 2, 3)
        assert Rs.shape == (B, N_ts, 3, 3)
        Omegas = Omegas.permute(1, 0, 2)
        assert Omegas.shape == (B, N_ts, 3)
        F_springs = F_springs.permute(1, 0, 2, 3)
        assert F_springs.shape == (B, N_ts, N_pts, 3)
        F_frictions = F_frictions.permute(1, 0, 2, 3)
        assert F_frictions.shape == (B, N_ts, N_pts, 3)

        return Xs, Xds, Rs, Omegas, F_springs, F_frictions

    def dphysics(self, z_grid, controls, joint_angles=None, state=None, friction=None):
        """
        Simulates the dynamics of the robot moving on the terrain.

        Parameters:
        - z_grid: Tensor of the height map (B, H, W).
        - controls: Tensor of control inputs (B, N, 2).
        - joint_angles: Tensor of joint angles (B, N, 4).
        - state: Tuple of the robot state (x, xd, R, omega).
        - stiffness: scalar or Tensor of the stiffness values at the robot points (B, H, W).
        - damping: scalar or Tensor of the damping values at the robot points (B, H, W).
        - friction: scalar or Tensor of the friction values at the robot points (B, H, W).

        Returns:
        - Tuple of the robot states and forces:
            - states: Tuple of the robot states (x, xd, R, omega).
            - forces: Tuple of the forces (F_springs, F_frictions).
        """
        # unpack config
        dt = self.dphys_cfg.dt
        T = self.dphys_cfg.traj_sim_time
        batch_size = z_grid.shape[0]

        # initial state
        if state is None:
            x = torch.tensor([0.0, 0.0, 0.0]).to(self.device).repeat(batch_size, 1)
            xd = torch.zeros_like(x); xd[:, 0] = controls[:, 0, 0]  # initial forward speed
            R = torch.eye(3).to(self.device).repeat(batch_size, 1, 1)
            omega = torch.zeros_like(x); omega[:, 2] = controls[:, 0, 1]  # initial rotational speed
            state = (x, xd, R, omega)

        # terrain properties
        friction = self.dphys_cfg.friction.repeat(batch_size, 1, 1) if friction is None else friction
        self.z_grid = z_grid.to(self.device)
        self.friction = friction.to(self.device)

        # start robot at the terrain height (not under or above the terrain)
        x = state[0]
        x_points = self.x_points.repeat(batch_size, 1, 1)
        x_points = x_points @ state[2].transpose(1, 2) + x.unsqueeze(1)
        z_interp = self.interpolate_grid(self.z_grid, x_points[..., 0], x_points[..., 1]).mean(dim=1, keepdim=True)
        x[..., 2:3] = z_interp

        N_ts = min(int(T / dt), controls.shape[1])
        B = state[0].shape[0]
        assert controls.shape == (B, N_ts, 2), f'Controls shape {controls.shape} != {(B, N_ts, 2)}'  # (B, N, 2), v, w
        self.controls = controls
        if joint_angles is None:
            joint_angles = torch.zeros((B, N_ts, 4), device=self.device)
        assert joint_angles.shape == (B, N_ts, 4), f'Joint angles shape {joint_angles.shape} != {(B, N_ts, 4)}'  # (B, N, 4), [fr, fl, rr, rl]
        self.joint_angles = joint_angles
        self.ts = self.ts[:N_ts]

        # dynamics of the rigid body
        Xs, Xds, Rs, Omegas, F_springs, F_frictions = self.integrator(state)

        # mg = k * delta_h, at equilibrium, delta_h = mg / k
        delta_h = self.dphys_cfg.robot_mass * self.dphys_cfg.gravity / (self.stiffness + 1e-6)
        # add the equilibrium height to the robot points along the z-axis of the robot
        Xs = Xs + Rs[:, :, :3, 2] * delta_h

        States = Xs, Xds, Rs, Omegas
        Forces = F_springs, F_frictions

        return States, Forces

    def forward(self, z_grid,
                controls, joint_angles=None,
                state=None, vis=False, friction=None):
        states, forces = self.dphysics(z_grid=z_grid,
                                       controls=controls, joint_angles=joint_angles, state=state,
                                       friction=friction)
        if vis:
            with torch.no_grad():
                self.visualize(states=states, z_grid=z_grid)  #, forces=forces)
        return states, forces

    def visualize(self, states, z_grid, forces=None, states_gt=None, friction=None):
        # visualize using mayavi
        from mayavi import mlab
        import os

        batch_i = np.random.choice(z_grid.shape[0])
        Xs, Xds, Rs, Omegas = [s.cpu().numpy() for s in states]
        x_grid_np, y_grid_np = self.dphys_cfg.x_grid.cpu().numpy(), self.dphys_cfg.y_grid.cpu().numpy()
        z_grid_np = z_grid[batch_i].cpu().numpy()
        x_points = self.dphys_cfg.robot_points.cpu().numpy()

        # set up the visualization
        mlab.figure(size=(1600, 800))
        mlab.plot3d(Xs[batch_i, :, 0], Xs[batch_i, :, 1], Xs[batch_i, :, 2], color=(0, 1, 0), line_width=2.0)
        if states_gt is not None:
            Xs_gt = states_gt[0].cpu().numpy()
            mlab.plot3d(Xs_gt[batch_i, :, 0], Xs_gt[batch_i, :, 1], Xs_gt[batch_i, :, 2], color=(0, 0, 1), line_width=2.0)
        # colorize the heightmap with friction values
        if friction is not None:
            friction_np = friction[batch_i].cpu().numpy()
            mlab.mesh(x_grid_np, y_grid_np, z_grid_np, scalars=friction_np, colormap='terrain', opacity=0.5)
            mlab.colorbar(orientation='horizontal', label_fmt='%.1f', nb_labels=5)
        else:
            mlab.mesh(x_grid_np, y_grid_np, z_grid_np, colormap='terrain', opacity=0.5)
            mlab.colorbar(orientation='horizontal', label_fmt='%.1f', nb_labels=5)
        mlab.surf(x_grid_np, y_grid_np, z_grid_np, representation='wireframe')
        visu_robot = mlab.points3d(x_points[:, 0], x_points[:, 1], x_points[:, 2],
                                   scale_factor=0.05, color=(0, 0, 0))
        if forces is not None:
            F_spring, F_friction = [f[batch_i].cpu().numpy() for f in forces]
            visu_Ns = mlab.quiver3d(x_points[:, 0], x_points[:, 1], x_points[:, 2],
                                    F_spring[0, :, 0], F_spring[0, :, 1], F_spring[0, :, 2],
                                    line_width=4, scale_factor=0.02, color=(0, 0, 1))
            visu_Frs = mlab.quiver3d(x_points[:, 0], x_points[:, 1], x_points[:, 2],
                                     F_friction[0, :, 0], F_friction[0, :, 1], F_friction[0, :, 2],
                                     line_width=4, scale_factor=0.02, color=(1, 0, 0))

        # set view point
        mlab.view(azimuth=95, elevation=80, distance=12.0, focalpoint=(0, 0, 0))

        # animate robot's motion and forces
        N_ts = Xs.shape[1]
        frame_i = 0
        path = os.path.join(os.path.dirname(__file__), '../../../gen/robot_control')
        os.makedirs(path, exist_ok=True)
        for t in range(0, N_ts, 10):
            # update the robot body points based on the joint angles
            joint_angles_t = self.joint_angles[batch_i, t][np.newaxis]
            x_points = self.update_joints(joint_angles_t).squeeze(0).cpu().numpy()
            # motion of point composed of cog motion and rotation of the rigid body
            x_points_t = x_points @ Rs[batch_i, t].T + Xs[batch_i, t][np.newaxis]
            visu_robot.mlab_source.set(x=x_points_t[:, 0], y=x_points_t[:, 1], z=x_points_t[:, 2])

            if forces is not None:
                F_spring_t, F_friction_t = F_spring[t], F_friction[t]
                visu_Ns.mlab_source.set(x=x_points_t[:, 0], y=x_points_t[:, 1], z=x_points_t[:, 2],
                                        u=F_spring_t[:, 0], v=F_spring_t[:, 1], w=F_spring_t[:, 2])
                visu_Frs.mlab_source.set(x=x_points_t[:, 0], y=x_points_t[:, 1], z=x_points_t[:, 2],
                                         u=F_friction_t[:, 0], v=F_friction_t[:, 1], w=F_friction_t[:, 2])

            mlab.savefig(f'{path}/{frame_i:04d}.png')
            frame_i += 1
        mlab.show()
