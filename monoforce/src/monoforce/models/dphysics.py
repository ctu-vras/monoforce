import torch
import numpy as np
from ..config import DPhysConfig


# constants
g = 9.81  # gravity, m/s^2

def interpolate_height(z_grid, x_query, y_query, d_max, grid_res):
    """
    Interpolates the height at the desired (x_query, y_query) coordinates.

    Parameters:
    - z_grid: Tensor of z values (heights) corresponding to the x and y coordinates (3D array), (B, H, W).
    - x_query: Tensor of desired x coordinates for interpolation (2D array), (B, N).
    - y_query: Tensor of desired y coordinates for interpolation (2D array), (B, N).
    - d_max: Maximum distance from the origin.
    - grid_res: Grid resolution.

    Returns:
    - Interpolated z values at the queried coordinates.
    """

    # Ensure inputs are tensors
    z_grid = torch.as_tensor(z_grid)
    x_query = torch.as_tensor(x_query)
    y_query = torch.as_tensor(y_query)

    # Get the grid dimensions
    B, H, W = z_grid.shape

    # Flatten the grid coordinates
    z_grid_flat = z_grid.reshape(B, -1)

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

    # Interpolate the z values
    z_query = (1 - x_f) * (1 - y_f) * z_grid_flat.gather(1, idx00) + \
              (1 - x_f) * y_f * z_grid_flat.gather(1, idx01) + \
              x_f * (1 - y_f) * z_grid_flat.gather(1, idx10) + \
              x_f * y_f * z_grid_flat.gather(1, idx11)

    return z_query


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
    assert mode in ['euler', 'rk4']
    if mode == 'euler':
        x = x + xd * dt
    elif mode == 'rk4':
        k1 = dt * xd
        k2 = dt * (xd + k1 / 2)
        k3 = dt * (xd + k2 / 2)
        k4 = dt * (xd + k3)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x


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


def surface_normals(z_grid, x_query, y_query, d_max, grid_res):
    """
    Computes the surface normals and tangents at the queried coordinates.

    Parameters:
    - z_grid: Tensor of z values (heights) corresponding to the x and y coordinates (3D array), (B, H, W).
    - x_query: Tensor of desired x coordinates for interpolation (2D array), (B, N).
    - y_query: Tensor of desired y coordinates for interpolation (2D array), (B, N).
    - d_max: Maximum distance from the origin.
    - grid_res: Grid resolution.

    Returns:
    - Surface normals and tangents at the queried coordinates.
    """
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

def forward_kinematics(x, xd, R, omega, x_points, xd_points,
                       z_grid, d_max, grid_res,
                       m, I_inv, mask_left, mask_right,
                       k_stiffness, k_damping, k_friction,
                       u_left, u_right):
    assert x.dim() == 2 and x.shape[1] == 3  # (B, 3)
    assert xd.dim() == 2 and xd.shape[1] == 3  # (B, 3)
    assert R.dim() == 3 and R.shape[-2:] == (3, 3)  # (B, 3, 3)
    assert x_points.dim() == 3 and x_points.shape[-1] == 3  # (B, N, 3)
    assert xd_points.dim() == 3 and xd_points.shape[-1] == 3  # (B, N, 3)
    assert mask_left.dim() == 2 and mask_left.shape[1] == x_points.shape[1]  # (B, N)
    assert mask_right.dim() == 2 and mask_right.shape[1] == x_points.shape[1]  # (B, N)
    # if scalar, convert to tensor
    if isinstance(u_left, (int, float)):
        u_left = torch.tensor([u_left], device=x.device)
    if isinstance(u_right, (int, float)):
        u_right = torch.tensor([u_right], device=x.device)
    assert u_left.dim() == 1  # scalar
    assert u_right.dim() == 1  # scalar
    assert z_grid.dim() == 3  # (B, H, W)
    assert I_inv.shape == (3, 3)  # (3, 3)
    B, n_pts, D = x_points.shape

    # check if the rigid body is in contact with the terrain
    z_points = interpolate_height(z_grid, x_points[..., 0], x_points[..., 1], d_max, grid_res)
    assert z_points.shape == (B, n_pts)
    dh_points = x_points[..., 2:3] - z_points.unsqueeze(-1)
    # in_contact = torch.sigmoid(-dh_points)
    in_contact = (dh_points <= 0.0).float()
    assert in_contact.shape == (B, n_pts, 1)

    # compute surface normals at the contact points
    n = surface_normals(z_grid, x_points[..., 0], x_points[..., 1], d_max, grid_res)
    assert n.shape == (B, n_pts, 3)

    # reaction at the contact points as spring-damper forces
    xd_points_n = (xd_points * n).sum(dim=-1, keepdims=True)  # normal velocity
    assert xd_points_n.shape == (B, n_pts, 1)
    F_spring = -torch.mul((k_stiffness * dh_points + k_damping * xd_points_n), n)  # F_s = -k * dh - b * v_n
    F_spring = torch.mul(F_spring, in_contact)
    assert F_spring.shape == (B, n_pts, 3)
    # limit the spring forces
    F_spring = torch.clamp(F_spring, min=0.0, max=2 * m * g)

    # friction forces: https://en.wikipedia.org/wiki/Friction
    N = torch.norm(F_spring, dim=-1, keepdim=True)
    xd_points_tau = xd_points - xd_points_n * n  # tangential velocities at the contact points
    tau = normailized(xd_points_tau)  # tangential directions of the velocities
    F_friction = -k_friction * N * tau  # F_fr = -k_fr * N * tau
    assert F_friction.shape == (B, n_pts, 3)

    # thrust forces: left and right
    thrust_dir = normailized(R @ torch.tensor([1.0, 0.0, 0.0], device=R.device))
    x_left = x_points[mask_left].mean(dim=0, keepdims=True)  # left thrust is applied at the mean of the left points
    x_right = x_points[mask_right].mean(dim=0, keepdims=True)  # right thrust is applied at the mean of the right points
    F_thrust_left = u_left.unsqueeze(1) * thrust_dir * in_contact[mask_left].mean()  # F_l = u_l * thrust_dir
    F_thrust_right = u_right.unsqueeze(1) * thrust_dir * in_contact[mask_right].mean()  # F_r = u_r * thrust_dir
    assert F_thrust_left.shape == (B, 3) == F_thrust_right.shape
    torque_left = torch.cross(x_left - x, F_thrust_left)  # M_l = (x_l - x) x F_l
    torque_right = torch.cross(x_right - x, F_thrust_right)  # M_r = (x_r - x) x F_r
    torque_thrust = torque_left + torque_right  # M_thrust = M_l + M_r
    assert torque_thrust.shape == (B, 3)

    # rigid body rotation: M = sum(r_i x F_i)
    torque = torch.sum(torch.cross(x_points - x.unsqueeze(1), F_spring + F_friction), dim=1) + torque_thrust
    omega_d = torque @ I_inv.transpose(0, 1)  # omega_d = I^(-1) M
    omega_skew = skew_symmetric(omega)  # omega_skew = [omega]_x
    dR = omega_skew @ R  # dR = [omega]_x R

    # motion of the cog
    F_grav = torch.tensor([[0.0, 0.0, -m * g]], device=x.device)  # F_grav = [0, 0, -m * g]
    F_cog = F_grav + F_spring.mean(dim=1) + F_friction.mean(dim=1) + F_thrust_left + F_thrust_right  # ma = sum(F_i)
    xdd = F_cog / m  # a = F / m
    assert xdd.shape == (B, 3)

    # motion of point composed of cog motion and rotation of the rigid body (Koenig's theorem in mechanics)
    xd_points = xd.unsqueeze(1) + torch.cross(omega.view(B, 1, 3), x_points - x.unsqueeze(1))
    assert xd_points.shape == (B, n_pts, 3)

    return xd, xdd, dR, omega_d, xd_points, F_spring, F_friction, F_thrust_left, F_thrust_right


def update_states(x, xd, xdd, R, dR, omega, omega_d, x_points, xd_points, dt):
    xd = integration_step(xd, xdd, dt)
    x = integration_step(x, xd, dt)
    x_points = integration_step(x_points, xd_points, dt)
    omega = integration_step(omega, omega_d, dt)
    R = integration_step(R, dR, dt)

    return x, xd, R, omega, x_points


def dphysics(z_grid, controls, state=None, robot_geometry=None, dphys_cfg=DPhysConfig()):
    # unpack config
    d_max = dphys_cfg.d_max
    grid_res = dphys_cfg.grid_res
    m = dphys_cfg.robot_mass
    device = z_grid.device
    I = torch.as_tensor(dphys_cfg.robot_I, device=device)
    k_stiffness = dphys_cfg.k_stiffness
    k_damping = dphys_cfg.k_damping
    k_friction = dphys_cfg.k_friction
    dt = dphys_cfg.dt
    T = dphys_cfg.traj_sim_time
    batch_size = z_grid.shape[0]
    if robot_geometry is None:
        mask_left = torch.as_tensor(dphys_cfg.robot_mask_left, device=device)
        mask_right = torch.as_tensor(dphys_cfg.robot_mask_right, device=device)
        mask_left = mask_left.repeat(batch_size, 1)
        mask_right = mask_right.repeat(batch_size, 1)
    else:
        mask_left, mask_right = robot_geometry

    if state is None:
        x = torch.tensor([[0.0, 0.0, 0.2]]).to(device).repeat(batch_size, 1)
        xd = torch.tensor([[0.0, 0.0, 0.0]]).to(device).repeat(batch_size, 1)
        R = torch.eye(3).to(device).repeat(batch_size, 1, 1)
        omega = torch.tensor([[0.0, 0.0, 0.0]]).to(device).repeat(batch_size, 1)
        x_points = torch.as_tensor(dphys_cfg.robot_points, device=device)
        x_points = x_points.repeat(batch_size, 1, 1)
        x_points = x_points @ R.transpose(1, 2) + x.unsqueeze(1)
        state = (x, xd, R, omega, x_points)

    N_ts = int(T / dt)
    B = state[0].shape[0]
    assert controls.shape == (B, N_ts, 2)  # for each time step, left and right thrust forces

    # state: x, xd, R, omega, x_points
    x, xd, R, omega, x_points = state
    xd_points = torch.zeros_like(x_points)

    I_inv = torch.inverse(I)
    if k_damping is None:
        k_damping = np.sqrt(4 * m * k_stiffness)  # critically damping

    # dynamics of the rigid body
    Xs, Xds, Rs, Omegas, Omega_ds, X_points = [], [], [], [], [], []
    F_springs, F_frictions, F_thrusts_left, F_thrusts_right = [], [], [], []
    ts = range(int(T / dt))
    B, N_ts, N_pts = x.shape[0], len(ts), x_points.shape[1]
    for i in ts:
        # control inputs
        u_left, u_right = controls[:, i, 0], controls[:, i, 1]  # thrust forces, Newtons or kg*m/s^2
        # forward kinematics
        (xd, xdd, dR, omega_d, xd_points,
         F_spring, F_friction, F_thrust_left, F_thrust_right) = forward_kinematics(x, xd, R, omega, x_points, xd_points,
                                                                                   z_grid, d_max, grid_res,
                                                                                   m, I_inv, mask_left, mask_right,
                                                                                   k_stiffness, k_damping, k_friction,
                                                                                   u_left, u_right)
        # update states: integration steps
        x, xd, R, omega, x_points = update_states(x, xd, xdd, R, dR, omega, omega_d, x_points, xd_points, dt)

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

    states = Xs, Xds, Rs, Omegas, X_points
    forces = F_springs, F_frictions, F_thrusts_left, F_thrusts_right

    return states, forces
