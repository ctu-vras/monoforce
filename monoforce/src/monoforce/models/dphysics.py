import torch
import numpy as np
from ..config import DPhysConfig


def normailized(x, eps=1e-6):
    """
    Normalizes the input tensor.

    Parameters:
    - x: Input tensor.
    - eps: Small value to avoid division by zero.

    Returns:
    - Normalized tensor.
    """
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


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

def vw_to_track_vel(v, w, r=1.0):
    # v: linear velocity, w: angular velocity, r: robot radius
    # v = (v_l + v_r) / 2
    # w = (v_l - v_r) / (2 * r)
    v_l = v + r * w
    v_r = v - r * w
    return v_l, v_r

class DPhysics(torch.nn.Module):
    def __init__(self, dphys_cfg=DPhysConfig(), device='cpu'):
        super(DPhysics, self).__init__()
        self.dphys_cfg = dphys_cfg
        self.device = device
        self.I = torch.as_tensor(dphys_cfg.robot_I, device=device)
        self.I_inv = torch.inverse(self.I)
        self.g = 9.81  # gravity, m/s^2
        self.robot_mask_left = torch.as_tensor(dphys_cfg.robot_mask_left, device=device)
        self.robot_mask_right = torch.as_tensor(dphys_cfg.robot_mask_right, device=device)

    def forward_kinematics(self, state, xd_points,
                           z_grid, stiffness, damping, friction,
                           m, mask_left, mask_right,
                           u_left, u_right):
        # unpack state
        x, xd, R, omega, x_points = state
        assert x.dim() == 2 and x.shape[1] == 3  # (B, 3)
        assert xd.dim() == 2 and xd.shape[1] == 3  # (B, 3)
        assert R.dim() == 3 and R.shape[-2:] == (3, 3)  # (B, 3, 3)
        assert x_points.dim() == 3 and x_points.shape[-1] == 3  # (B, N, 3)
        assert xd_points.dim() == 3 and xd_points.shape[-1] == 3  # (B, N, 3)
        assert mask_left.dim() == 1 and mask_left.shape[0] == x_points.shape[1]  # (N,)
        assert mask_right.dim() == 1 and mask_right.shape[0] == x_points.shape[1]  # (N,)
        # if scalar, convert to tensor
        if isinstance(u_left, (int, float)):
            u_left = torch.tensor([u_left], device=self.device)
        if isinstance(u_right, (int, float)):
            u_right = torch.tensor([u_right], device=self.device)
        assert u_left.dim() == 1  # scalar
        assert u_right.dim() == 1  # scalar
        assert z_grid.dim() == 3  # (B, H, W)
        B, n_pts, D = x_points.shape

        # compute the terrain properties at the robot points
        z_points = self.interpolate_grid(z_grid, x_points[..., 0], x_points[..., 1]).unsqueeze(-1)
        assert z_points.shape == (B, n_pts, 1)
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

        # compute surface normals at the contact points
        n = self.surface_normals(z_grid, x_points[..., 0], x_points[..., 1])
        assert n.shape == (B, n_pts, 3)

        # reaction at the contact points as spring-damper forces
        xd_points_n = (xd_points * n).sum(dim=-1, keepdims=True)  # normal velocity
        assert xd_points_n.shape == (B, n_pts, 1)
        F_spring = -torch.mul((stiffness_points * dh_points + damping_points * xd_points_n), n)  # F_s = -k * dh - b * v_n
        F_spring = torch.mul(F_spring, in_contact)
        assert F_spring.shape == (B, n_pts, 3)

        # friction forces: https://en.wikipedia.org/wiki/Friction
        N = torch.norm(F_spring, dim=2)
        xd_points_tau = xd_points - xd_points_n * n  # tangential velocities at the contact points
        tau = normailized(xd_points_tau)  # tangential directions of the velocities
        F_friction = -friction_points * N.unsqueeze(2) * tau  # F_fr = -k_fr * N * tau
        assert F_friction.shape == (B, n_pts, 3)

        # thrust forces: left and right
        thrust_dir = normailized(R @ torch.tensor([1.0, 0.0, 0.0], device=self.device))
        x_left = x_points[:, mask_left].mean(dim=1)  # left thrust is applied at the mean of the left points
        x_right = x_points[:, mask_right].mean(dim=1)  # right thrust is applied at the mean of the right points
        xd_left = xd_points[:, mask_left].mean(dim=1)  # mean velocity of the left points
        xd_right = xd_points[:, mask_right].mean(dim=1)  # mean velocity of the right points
        assert x_left.shape == x_right.shape == xd_left.shape == xd_right.shape == (B, 3)

        # compute thrust forces in a way that left part of the robot moves with the desired velocity u_left and right part with u_right
        v_l = (xd_left * thrust_dir).sum(dim=-1)  # v_l = xd_l . thrust_dir
        v_r = (xd_right * thrust_dir).sum(dim=-1)  # v_r = xd_r . thrust_dir
        F_thrust_left = (N.mean(dim=1) * (u_left - v_l)).unsqueeze(1) * thrust_dir * in_contact[:, mask_left].mean(dim=1)  # F_l = N * (u_l - v_l) * thrust_dir
        F_thrust_right = (N.mean(dim=1) * (u_right - v_r)).unsqueeze(1) * thrust_dir * in_contact[:, mask_right].mean(dim=1)  # F_r = N * (u_r - v_r) * thrust_dir

        assert F_thrust_left.shape == (B, 3) == F_thrust_right.shape
        torque_left = torch.cross(x_left - x, F_thrust_left)  # M_l = (x_l - x) x F_l
        torque_right = torch.cross(x_right - x, F_thrust_right)  # M_r = (x_r - x) x F_r
        torque_thrust = torque_left + torque_right  # M_thrust = M_l + M_r
        assert torque_thrust.shape == (B, 3)

        # rigid body rotation: M = sum(r_i x F_i)
        torque = torch.sum(torch.cross(x_points - x.unsqueeze(1), F_spring + F_friction), dim=1) + torque_thrust
        omega_d = torque @ self.I_inv.transpose(0, 1)  # omega_d = I^(-1) M
        Omega_skew = skew_symmetric(omega)  # Omega_skew = [omega]_x
        dR = Omega_skew @ R  # dR = [omega]_x R

        # motion of the cog
        F_grav = torch.tensor([[0.0, 0.0, -m * self.g]], device=self.device)  # F_grav = [0, 0, -m * g]
        F_cog = F_grav + F_spring.mean(dim=1) + F_friction.mean(dim=1) + F_thrust_left + F_thrust_right  # ma = sum(F_i)
        xdd = F_cog / m  # a = F / m
        assert xdd.shape == (B, 3)

        # motion of point composed of cog motion and rotation of the rigid body (Koenig's theorem in mechanics)
        xd_points = xd.unsqueeze(1) + torch.cross(omega.unsqueeze(1), x_points - x.unsqueeze(1))
        assert xd_points.shape == (B, n_pts, 3)

        dstate = (xd, xdd, dR, omega_d, xd_points)
        forces = (F_spring, F_friction, F_thrust_left, F_thrust_right)

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
        theta = torch.norm(omega, dim=-1, keepdim=True).unsqueeze(-1) * dt

        # Normalize the angular velocities
        Omega_x_norm = Omega_x / (theta / dt + eps)

        # Rodrigues' formula: R_new = R * (I + |Omega_x| * sin(theta) + |Omega_x|^2 * (1 - cos(theta)))
        I = torch.eye(3).to(R.device)
        R_new = R @ (I + Omega_x_norm * torch.sin(theta) + Omega_x_norm @ Omega_x_norm * (1 - torch.cos(theta)))

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

    def surface_normals(self, z_grid, x_query, y_query):
        """
        Computes the surface normals and tangents at the queried coordinates.

        Parameters:
        - z_grid: Tensor of z values (heights) corresponding to the x and y coordinates (3D array), (B, H, W).
        - x_query: Tensor of desired x coordinates for interpolation (2D array), (B, N).
        - y_query: Tensor of desired y coordinates for interpolation (2D array), (B, N).

        Returns:
        - Surface normals at the queried coordinates.
        """
        # unpack config
        d_max = self.dphys_cfg.d_max
        grid_res = self.dphys_cfg.grid_res

        # Ensure inputs are tensors
        z_grid = torch.as_tensor(z_grid)
        x_query = torch.as_tensor(x_query)
        y_query = torch.as_tensor(y_query)

        # Get the grid dimensions
        B, H, W = z_grid.shape

        # Compute the indices of the grid points surrounding the query points
        x_i = torch.clamp(((x_query + d_max) / grid_res).long(), 0, W - 2)
        y_i = torch.clamp(((y_query + d_max) / grid_res).long(), 0, H - 2)

        # Compute the fractional part of the indices
        x_f = (x_query + d_max) / grid_res - x_i.float()
        y_f = (y_query + d_max) / grid_res - y_i.float()

        # Compute the indices of the grid points
        idx00 = x_i + W * y_i
        idx01 = x_i + W * (y_i + 1)
        idx10 = (x_i + 1) + W * y_i
        idx11 = (x_i + 1) + W * (y_i + 1)

        # Interpolate the z values
        z_grid_flat = z_grid.reshape(B, -1)
        z00 = z_grid_flat.gather(1, idx00)
        z01 = z_grid_flat.gather(1, idx01)
        z10 = z_grid_flat.gather(1, idx10)
        z11 = z_grid_flat.gather(1, idx11)

        # Compute the surface normals
        dz_dx = (z10 - z00) * (1 - y_f) + (z11 - z01) * y_f
        dz_dy = (z01 - z00) * (1 - x_f) + (z11 - z10) * x_f
        n = torch.stack([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=-1)  # n = [-dz/dx, -dz/dy, 1]
        n = normailized(n)

        return n

    def interpolate_grid(self, grid, x_query, y_query):
        """
        Interpolates the height at the desired (x_query, y_query) coordinates.

        Parameters:
        - grid: Tensor of grid values corresponding to the x and y coordinates (3D array), (B, H, W).
        - x_query: Tensor of desired x coordinates for interpolation (2D array), (B, N).
        - y_query: Tensor of desired y coordinates for interpolation (2D array), (B, N).

        Returns:
        - Interpolated grid values at the queried coordinates.
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
        x_i = torch.clamp(((x_query_flat + d_max) / grid_res).long(), 0, W - 2)
        y_i = torch.clamp(((y_query_flat + d_max) / grid_res).long(), 0, H - 2)

        # Compute the fractional part of the indices
        x_f = (x_query_flat + d_max) / grid_res - x_i.float()
        y_f = (y_query_flat + d_max) / grid_res - y_i.float()

        # Compute the indices of the grid points
        idx00 = x_i + W * y_i
        idx01 = x_i + W * (y_i + 1)
        idx10 = (x_i + 1) + W * y_i
        idx11 = (x_i + 1) + W * (y_i + 1)

        # Interpolate the z values (linear interpolation)
        z_query = (1 - x_f) * (1 - y_f) * z_grid_flat.gather(1, idx00) + \
                  (1 - x_f) * y_f * z_grid_flat.gather(1, idx01) + \
                  x_f * (1 - y_f) * z_grid_flat.gather(1, idx10) + \
                  x_f * y_f * z_grid_flat.gather(1, idx11)

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

        # robot geometry masks for left and right thrust points
        mask_left = self.robot_mask_left
        mask_right = self.robot_mask_right

        # initial state
        if state is None:
            x = torch.tensor([0.0, 0.0, 0.2]).to(device).repeat(batch_size, 1)
            xd = torch.zeros_like(x)
            R = torch.eye(3).to(device).repeat(batch_size, 1, 1)
            omega = torch.zeros_like(x)
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
        assert controls.shape == (B, N_ts, 2)  # for each time step, left and right thrust forces

        # TODO: there is some bug, had to transpose grid map
        z_grid = z_grid.transpose(1, 2)  # (B, H, W) -> (B, W, H)
        stiffness = stiffness.transpose(1, 2) if not isinstance(stiffness, (int, float)) else stiffness
        damping = damping.transpose(1, 2) if not isinstance(damping, (int, float)) else damping
        friction = friction.transpose(1, 2) if not isinstance(friction, (int, float)) else friction

        # state: x, xd, R, omega, x_points
        x, xd, R, omega, x_points = state
        xd_points = torch.zeros_like(x_points)

        # dynamics of the rigid body
        Xs, Xds, Rs, Omegas, Omega_ds, X_points = [], [], [], [], [], []
        F_springs, F_frictions, F_thrusts_left, F_thrusts_right = [], [], [], []
        ts = range(N_ts)
        B, N_ts, N_pts = x.shape[0], len(ts), x_points.shape[1]
        for t in ts:
            # control inputs
            u_left, u_right = controls[:, t, 0], controls[:, t, 1]  # thrust forces, Newtons or kg*m/s^2
            # forward kinematics
            dstate, forces = self.forward_kinematics(state=state, xd_points=xd_points,
                                                     z_grid=z_grid,
                                                     stiffness=stiffness, damping=damping, friction=friction,
                                                     m=self.dphys_cfg.robot_mass,
                                                     mask_left=mask_left, mask_right=mask_right,
                                                     u_left=u_left, u_right=u_right,)
            # update state: integration steps
            state = self.update_state(state, dstate, dt)

            # unpack state, its differential, and forces
            x, xd, R, omega, x_points = state
            _, xdd, dR, omega_d, xd_points = dstate
            F_spring, F_friction, F_thrust_left, F_thrust_right = forces

            # save states
            Xs.append(x)
            Xds.append(xd)
            Rs.append(R)
            Omegas.append(omega)
            X_points.append(x_points)

            # save forces
            F_springs.append(F_spring)
            F_frictions.append(F_friction)
            F_thrusts_left.append(F_thrust_left)
            F_thrusts_right.append(F_thrust_right)

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
        F_thrusts_left = torch.stack(F_thrusts_left).transpose(1, 0)
        assert F_thrusts_left.shape == (B, N_ts, 3)
        F_thrusts_right = torch.stack(F_thrusts_right).transpose(1, 0)
        assert F_thrusts_right.shape == (B, N_ts, 3)

        States = Xs, Xds, Rs, Omegas, X_points
        Forces = F_springs, F_frictions, F_thrusts_left, F_thrusts_right

        return States, Forces

    def forward(self, z_grid, controls, state=None, **kwargs):
        return self.dphysics(z_grid, controls, state, **kwargs)
