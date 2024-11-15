import torch
from ..dphys_config import DPhysConfig


def normalized(x, eps=1e-6):
    """
    Normalizes the input tensor.

    Parameters:
    - x: Input tensor.
    - eps: Small value to avoid division by zero.

    Returns:
    - Normalized tensor.
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
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

def generate_control_inputs(n_trajs=10,
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


class DPhysics(torch.nn.Module):
    def __init__(self, dphys_cfg=DPhysConfig(), device='cpu'):
        super(DPhysics, self).__init__()
        self.dphys_cfg = dphys_cfg
        self.device = device
        self.I = torch.as_tensor(self.dphys_cfg.robot_I, device=device)  # 3x3 inertia tensor, kg*m^2
        self.I_inv = torch.inverse(self.I)  # inverse of the inertia tensor

    def forward_kinematics(self, state, xd_points,
                           z_grid, stiffness, damping, friction,
                           driving_parts,
                           controls):
        # unpack state
        x, xd, R, omega, x_points = state
        assert x.dim() == 2 and x.shape[1] == 3  # (B, 3)
        assert xd.dim() == 2 and xd.shape[1] == 3  # (B, 3)
        assert R.dim() == 3 and R.shape[-2:] == (3, 3)  # (B, 3, 3)
        assert x_points.dim() == 3 and x_points.shape[-1] == 3  # (B, N, 3)
        assert xd_points.dim() == 3 and xd_points.shape[-1] == 3  # (B, N, 3)
        for p in driving_parts:
            assert p.dim() == 1 and p.shape[0] == x_points.shape[1]  # (N,)
        assert controls.dim() == 2 and controls.shape[0] == x.shape[0]  # (B, 2)
        assert controls.shape[1] == 2  # linear and angular velocities
        assert z_grid.dim() == 3  # (B, H, W)
        B, n_pts, D = x_points.shape

        # compute the terrain properties at the robot points
        z_points, n = self.interpolate_grid(z_grid, x_points[..., 0], x_points[..., 1], return_normals=True)
        z_points = z_points.unsqueeze(-1)
        assert z_points.shape == (B, n_pts, 1)
        assert n.shape == (B, n_pts, 3)
        if not isinstance(stiffness, (int, float)):
            stiffness_points = self.interpolate_grid(stiffness, x_points[..., 0], x_points[..., 1]).unsqueeze(-1)
            assert stiffness_points.shape == (B, n_pts, 1)
        else:
            stiffness_points = stiffness
        if not isinstance(damping, (int, float)):
            damping_points = self.interpolate_grid(damping, x_points[..., 0], x_points[..., 1]).unsqueeze(-1)
            assert damping_points.shape == (B, n_pts, 1)
        else:
            damping_points = damping
        if not isinstance(friction, (int, float)):
            friction_points = self.interpolate_grid(friction, x_points[..., 0], x_points[..., 1]).unsqueeze(-1)
            assert friction_points.shape == (B, n_pts, 1)
        else:
            friction_points = friction

        # check if the rigid body is in contact with the terrain
        dh_points = x_points[..., 2:3] - z_points
        on_grid = (x_points[..., 0:1] >= -self.dphys_cfg.d_max) & (x_points[..., 0:1] <= self.dphys_cfg.d_max) & \
                    (x_points[..., 1:2] >= -self.dphys_cfg.d_max) & (x_points[..., 1:2] <= self.dphys_cfg.d_max)
        in_contact = ((dh_points <= 0.0) & on_grid).float()
        assert in_contact.shape == (B, n_pts, 1)

        # reaction at the contact points as spring-damper forces
        xd_points_n = (xd_points * n).sum(dim=-1, keepdims=True)  # normal velocity
        assert xd_points_n.shape == (B, n_pts, 1)
        F_spring = -torch.mul((stiffness_points * dh_points + damping_points * xd_points_n), n)  # F_s = -k * dh - b * v_n
        F_spring = torch.mul(F_spring, in_contact) / n_pts  # apply forces only at the contact points
        assert F_spring.shape == (B, n_pts, 3)

        # friction forces: https://en.wikipedia.org/wiki/Friction
        thrust_dir = normalized(R[..., 0])  # direction of the thrust
        N = torch.norm(F_spring, dim=2)  # normal force magnitude at the contact points
        m, g = self.dphys_cfg.robot_mass, self.dphys_cfg.gravity
        F_friction = torch.zeros_like(F_spring)  # initialize friction forces
        track_vels = vw_to_track_vels(v=controls[:, 0], w=controls[:, 1],
                                      robot_size=self.dphys_cfg.robot_size, n_tracks=len(driving_parts))
        assert track_vels.shape == (B, len(driving_parts))
        for i in range(len(driving_parts)):
            u = track_vels[:, i].unsqueeze(1)  # control input
            v_cmd = u * thrust_dir  # commanded velocity
            mask = driving_parts[i]
            # F_fr = -mu * N * tanh(v_cmd - xd_points)  # tracks friction forces
            dv = v_cmd.unsqueeze(1) - xd_points
            dv_n = (dv * n).sum(dim=-1, keepdims=True)  # normal component of the relative velocity
            dv_tau = dv - dv_n * n  # tangential component of the relative velocity
            F_friction[:, mask] = (friction_points * N.unsqueeze(2) * torch.tanh(dv_tau))[:, mask]
        assert F_friction.shape == (B, n_pts, 3)

        # rigid body rotation: M = sum(r_i x F_i)
        torque = torch.sum(torch.linalg.cross(x_points - x.unsqueeze(1), F_spring + F_friction), dim=1)
        omega_d = torque @ self.I_inv.transpose(0, 1)  # omega_d = I^(-1) M
        omega_d = torch.clamp(omega_d, min=-self.dphys_cfg.omega_max, max=self.dphys_cfg.omega_max)
        Omega_skew = skew_symmetric(omega)  # Omega_skew = [omega]_x
        dR = Omega_skew @ R  # dR = [omega]_x R

        # motion of the cog
        F_grav = torch.tensor([[0.0, 0.0, -m * g]], device=self.device)  # F_grav = [0, 0, -m * g]
        F_cog = F_grav + F_spring.sum(dim=1) + F_friction.sum(dim=1)  # ma = sum(F_i)
        xdd = F_cog / m  # a = F / m
        assert xdd.shape == (B, 3)

        # motion of point composed of cog motion and rotation of the rigid body
        # Koenig's theorem in mechanics: v_i = v_cog + omega x (r_i - r_cog)
        xd_points = xd.unsqueeze(1) + torch.linalg.cross(omega.unsqueeze(1), x_points - x.unsqueeze(1))
        assert xd_points.shape == (B, n_pts, 3)

        dstate = (xd, xdd, dR, omega_d, xd_points)
        forces = (F_spring, F_friction)

        return dstate, forces

    def update_state(self, state, dstate, dt):
        """
        Integrates the states of the rigid body for the next time step.
        """
        x, xd, R, omega, x_points = state
        _, xdd, dR, omega_d, xd_points = dstate

        xd = self.integration_step(xd, xdd, dt, mode=self.dphys_cfg.integration_mode)
        x = self.integration_step(x, xd, dt, mode=self.dphys_cfg.integration_mode)
        x_points = self.integration_step(x_points, xd_points, dt, mode=self.dphys_cfg.integration_mode)
        omega = self.integration_step(omega, omega_d, dt, mode=self.dphys_cfg.integration_mode)
        # R = self.integration_step(R, dR, dt, mode=self.dphys_cfg.integration_mode)
        R = self.integrate_rotation(R, omega, dt)

        state = (x, xd, R, omega, x_points)

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
        # R_new = R @ (I + torch.sin(theta * dt) * Omega_x / theta + (1 - torch.cos(theta * dt)) * (Omega_x @ Omega_x) / (theta ** 2))

        return R_new

    @staticmethod
    def integration_step(x, xd, dt, mode='rk4'):
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
        elif mode == 'rk2':
            k1 = dt * xd
            k2 = dt * (xd + k1)
            x = x + k2 / 2
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
        x_f = (x_query_flat + d_max) / grid_res - x_i.float()
        y_f = (y_query_flat + d_max) / grid_res - y_i.float()

        # Compute the indices of the grid points
        idx00 = y_i + H * x_i
        idx01 = y_i + H * (x_i + 1)
        idx10 = (y_i + 1) + H * x_i
        idx11 = (y_i + 1) + H * (x_i + 1)
        # Clamp the indices to avoid out-of-bound errors
        idx00 = torch.clamp(idx00, 0, H * W - 1)
        idx01 = torch.clamp(idx01, 0, H * W - 1)
        idx10 = torch.clamp(idx10, 0, H * W - 1)
        idx11 = torch.clamp(idx11, 0, H * W - 1)

        # Interpolate the z values (linear interpolation)
        z_query = (1 - x_f) * (1 - y_f) * z_grid_flat.gather(1, idx00) + \
                  (1 - x_f) * y_f * z_grid_flat.gather(1, idx01) + \
                  x_f * (1 - y_f) * z_grid_flat.gather(1, idx10) + \
                  x_f * y_f * z_grid_flat.gather(1, idx11)

        if return_normals:
            # Estimate normals
            z00 = z_query
            z10 = self.interpolate_grid(grid, x_query + grid_res, y_query)
            z01 = self.interpolate_grid(grid, x_query, y_query + grid_res)
            dz_dx = (z10 - z00) / grid_res
            dz_dy = (z01 - z00) / grid_res
            n = torch.stack([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=-1)
            n = normalized(n)
            return z_query, n

        return z_query

    def dphysics(self, z_grid, controls, state=None, stiffness=None, damping=None, friction=None):
        """
        Simulates the dynamics of the robot moving on the terrain.

        Parameters:
        - z_grid: Tensor of the height map (B, H, W).
        - controls: Tensor of control inputs (B, N, 2).
        - state: Tuple of the robot state (x, xd, R, omega, x_points).
        - stiffness: scalar or Tensor of the stiffness values at the robot points (B, H, W).
        - damping: scalar or Tensor of the damping values at the robot points (B, H, W).
        - friction: scalar or Tensor of the friction values at the robot points (B, H, W).

        Returns:
        - Tuple of the robot states and forces:
            - states: Tuple of the robot states (x, xd, R, omega, x_points).
            - forces: Tuple of the forces (F_springs, F_frictions, F_thrusts_left, F_thrusts_right).
        """
        # unpack config
        device = self.device
        dt = self.dphys_cfg.dt
        T = self.dphys_cfg.traj_sim_time
        batch_size = z_grid.shape[0]

        # initial state
        if state is None:
            x = torch.tensor([0.0, 0.0, 0.0]).to(device).repeat(batch_size, 1)
            xd = torch.zeros_like(x); xd[:, 0] = controls[:, 0, 0]  # initial forward speed
            R = torch.eye(3).to(device).repeat(batch_size, 1, 1)
            omega = torch.zeros_like(x); omega[:, 2] = controls[:, 0, 1]  # initial rotational speed
            x_points = torch.as_tensor(self.dphys_cfg.robot_points, device=device)
            x_points = x_points.repeat(batch_size, 1, 1)
            x_points = x_points @ R.transpose(1, 2) + x.unsqueeze(1)
            state = (x, xd, R, omega, x_points)

        # terrain properties
        stiffness = self.dphys_cfg.k_stiffness if stiffness is None else stiffness
        damping = self.dphys_cfg.k_damping if damping is None else damping
        friction = self.dphys_cfg.k_friction if friction is None else friction

        N_ts = min(int(T / dt), controls.shape[1])
        B = state[0].shape[0]
        # for each trajectory and time step driving parts are being controlled
        assert controls.shape == (B, N_ts, 2), f'Its shape {controls.shape} != {(B, N_ts, 2)}'

        # state: x, xd, R, omega, x_points
        x, xd, R, omega, x_points = state
        # Koenig's theorem in mechanics: v_i = v_cog + omega x (r_i - r_cog)
        xd_points = xd.unsqueeze(1) + torch.linalg.cross(omega.unsqueeze(1), x_points - x.unsqueeze(1))

        # dynamics of the rigid body
        Xs, Xds, Rs, Omegas, Omega_ds, X_points = [], [], [], [], [], []
        F_springs, F_frictions = [], []
        ts = range(N_ts)
        B, N_ts, N_pts = x.shape[0], len(ts), x_points.shape[1]
        for t in ts:
            # forward kinematics
            dstate, forces = self.forward_kinematics(state=state, xd_points=xd_points,
                                                     z_grid=z_grid,
                                                     stiffness=stiffness, damping=damping, friction=friction,
                                                     driving_parts=self.dphys_cfg.driving_parts,
                                                     controls=controls[:, t])
            # update state: integration steps
            state = self.update_state(state, dstate, dt)

            # unpack state, its differential, and forces
            x, xd, R, omega, x_points = state
            _, xdd, dR, omega_d, xd_points = dstate
            F_spring, F_friction = forces

            # save states
            Xs.append(x)
            Xds.append(xd)
            Rs.append(R)
            Omegas.append(omega)
            X_points.append(x_points)

            # save forces
            F_springs.append(F_spring)
            F_frictions.append(F_friction)

        # to tensors
        Xs = torch.stack(Xs).transpose(1, 0)
        assert Xs.shape == (B, N_ts, 3)
        Xds = torch.stack(Xds).transpose(1, 0)
        assert Xds.shape == (B, N_ts, 3)
        Rs = torch.stack(Rs).transpose(1, 0)
        assert Rs.shape == (B, N_ts, 3, 3)
        Omegas = torch.stack(Omegas).transpose(1, 0)
        assert Omegas.shape == (B, N_ts, 3)
        X_points = torch.stack(X_points).transpose(1, 0)
        assert X_points.shape == (B, N_ts, N_pts, 3)
        F_springs = torch.stack(F_springs).transpose(1, 0)
        assert F_springs.shape == (B, N_ts, N_pts, 3)
        F_frictions = torch.stack(F_frictions).transpose(1, 0)
        assert F_frictions.shape == (B, N_ts, N_pts, 3)

        States = Xs, Xds, Rs, Omegas, X_points
        Forces = F_springs, F_frictions

        return States, Forces

    def forward(self, z_grid, controls, state=None, **kwargs):
        return self.dphysics(z_grid, controls, state, **kwargs)
