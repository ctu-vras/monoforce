#!/usr/bin/env python

import sys
sys.path.append('../src')
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from monoforce.models.traj_predictor.dphys_config import DPhysConfig
from monoforce.models.traj_predictor.dphysics import DPhysics, generate_controls
import matplotlib.pyplot as plt


# simulation parameters
dphys_cfg = DPhysConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def motion():
    # instantiate the simulator
    dphysics = DPhysics(dphys_cfg, device=device)

    # control inputs: linear velocity and angular velocity, v in m/s, w in rad/s
    controls = torch.stack([
        torch.tensor([[1.0, 0.6]] * int(dphys_cfg.traj_sim_time / dphys_cfg.dt)),  # [v] m/s, [w] rad/s for each time step
    ]).to(device)
    B, N_ts, _ = controls.shape
    assert controls.shape == (B, N_ts, 2), f'controls shape: {controls.shape}'

    # joint angles: [theta_fl, theta_fr, theta_rl, theta_rr]
    joint_angles = torch.stack([
        torch.linspace(-np.pi, np.pi, N_ts).repeat(B, 1),  # front left
        torch.linspace(-np.pi, np.pi, N_ts).repeat(B, 1),  # front right
        -torch.linspace(-np.pi, np.pi, N_ts).repeat(B, 1),  # rear left
        -torch.linspace(-np.pi, np.pi, N_ts).repeat(B, 1),  # rear right
    ], dim=-1).to(device)
    assert joint_angles.shape == (B, N_ts, 4), f'joint_angles shape: {joint_angles.shape}'

    # initial state
    x = torch.tensor([[-1.0, 0.0, 0.0]], device=device).repeat(B, 1)
    assert x.shape == (B, 3)
    xd = torch.zeros_like(x)
    assert xd.shape == (B, 3)
    rs = torch.tensor(Rotation.from_euler('z', [0.0]).as_matrix(), dtype=torch.float32, device=device)
    assert rs.shape == (B, 3, 3)
    omega = torch.zeros_like(x)
    assert omega.shape == (B, 3)
    state0 = (x, xd, rs, omega)

    # heightmap defining the terrain
    x_grid, y_grid = dphys_cfg.x_grid, dphys_cfg.y_grid
    # z_grid = torch.sin(x_grid) * torch.cos(y_grid)
    z_grid = torch.exp(-(x_grid - 2) ** 2 / 4) * torch.exp(-(y_grid - 0) ** 2 / 2)
    # z_grid = torch.zeros_like(x_grid)
    # z_grid[80:81, 0:100] = 1.0  # add a wall
    z_grid += 0.02 * torch.randn_like(z_grid)  # add noise

    x_grid, y_grid, z_grid = x_grid.to(device), y_grid.to(device), z_grid.to(device)
    friction = dphys_cfg.friction
    friction = friction.to(device)

    # repeat the heightmap for each rigid body
    x_grid = x_grid.repeat(x.shape[0], 1, 1)
    y_grid = y_grid.repeat(x.shape[0], 1, 1)
    z_grid = z_grid.repeat(x.shape[0], 1, 1)
    friction = friction.repeat(x.shape[0], 1, 1)
    H, W = int(2 * dphys_cfg.d_max / dphys_cfg.grid_res), int(2 * dphys_cfg.d_max / dphys_cfg.grid_res)
    assert x_grid.shape == (B, H, W)
    assert y_grid.shape == (B, H, W)
    assert z_grid.shape == (B, H, W)
    assert friction.shape == (B, H, W)

    # simulate the rigid body dynamics
    states, forces = dphysics(z_grid=z_grid,
                              controls=controls,
                              joint_angles=joint_angles,
                              state=state0,
                              friction=friction, vis=True)

def shoot_multiple():
    from time import time
    from monoforce.vis import set_axes_equal

    # simulation parameters
    dt = dphys_cfg.dt
    T = dphys_cfg.traj_sim_time
    num_trajs = dphys_cfg.n_sim_trajs
    vel_max, omega_max = dphys_cfg.vel_max, dphys_cfg.omega_max

    # instantiate the simulator
    dphysics = DPhysics(dphys_cfg, device=device)

    # initial state
    x = torch.tensor([[0.0, 0.0, 0.0]], device=device).repeat(num_trajs, 1)
    xd = torch.zeros_like(x)
    R = torch.eye(3, device=device).repeat(x.shape[0], 1, 1)
    # R = torch.tensor(Rotation.from_euler('z', np.pi/6).as_matrix(), dtype=torch.float32, device=device).repeat(num_trajs, 1, 1)
    omega = torch.zeros_like(x)

    # terrain properties
    x_grid, y_grid = dphys_cfg.x_grid, dphys_cfg.y_grid
    z_grid = torch.exp(-(x_grid - 2) ** 2 / 4) * torch.exp(-(y_grid - 0) ** 2 / 2)
    # z_grid = torch.sin(x_grid) * torch.cos(y_grid)
    # z_grid = torch.zeros_like(x_grid)

    # repeat the heightmap for each rigid body
    x_grid = x_grid.repeat(num_trajs, 1, 1)
    y_grid = y_grid.repeat(num_trajs, 1, 1)
    z_grid = z_grid.repeat(num_trajs, 1, 1)

    # control inputs in m/s and rad/s
    controls_front, _ = generate_controls(n_trajs=num_trajs // 2,
                                          v_range=(vel_max / 2, vel_max), w_range=(-omega_max, omega_max),
                                          time_horizon=T, dt=dt)
    controls_back, _ = generate_controls(n_trajs=num_trajs // 2,
                                         v_range=(-vel_max, -vel_max / 2), w_range=(-omega_max, omega_max),
                                         time_horizon=T, dt=dt)
    controls = torch.cat([controls_front, controls_back], dim=0)
    # controls = torch.ones_like(controls)
    controls = torch.as_tensor(controls, dtype=torch.float32, device=device)

    # initial state
    state0 = (x, xd, R, omega)

    # put tensors to device
    state0 = tuple([s.to(device) for s in state0])
    z_grid = z_grid.to(device)
    controls = controls.to(device)

    # simulate the rigid body dynamics
    with torch.no_grad():
        t0 = time()
        states, forces = dphysics(z_grid=z_grid, controls=controls, state=state0, vis=False)
        t1 = time()
        Xs, Xds, Rs, Omegas = states
        print(f'Simulation took {(t1-t0):.3f} [sec] on device: {device}')

    # visualize
    with torch.no_grad():
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # plot heightmap
        ax.plot_surface(x_grid[0].cpu().numpy(), y_grid[0].cpu().numpy(), z_grid[0].cpu().numpy(), alpha=0.6, cmap='terrain')
        set_axes_equal(ax)
        for i in range(num_trajs):
            X = Xs[i].cpu().numpy()
            ax.plot(X[:, 0], X[:, 1], X[:, 2], label=f'Traj {i}', c='g')
        ax.set_title(f'Simulation of {num_trajs} trajs (T={T} [sec] long) took {(t1-t0):.3f} [sec] on device: {device}')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        plt.show()


def main():
    motion()
    shoot_multiple()


if __name__ == '__main__':
    main()
