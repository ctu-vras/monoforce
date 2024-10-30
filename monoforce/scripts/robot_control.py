#!/usr/bin/env python

import sys
sys.path.append('../src')
import torch
import numpy as np
from monoforce.dphys_config import DPhysConfig
from monoforce.models.dphysics import DPhysics, generate_control_inputs
from monoforce.vis import setup_visualization, animate_trajectory
import matplotlib.pyplot as plt


# simulation parameters
robot = 'tradr2'
dphys_cfg = DPhysConfig(robot=robot)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def motion():
    from scipy.spatial.transform import Rotation
    # simulation parameters
    dt = dphys_cfg.dt
    T = dphys_cfg.traj_sim_time

    # instantiate the simulator
    dphysics = DPhysics(dphys_cfg, device=device)

    # control inputs: track vels in m/s
    if dphys_cfg.robot in ['marv', 'husky']:
        # fl, fr, rl, rr
        controls = torch.stack([
            torch.tensor([[0.5, 0.8, 0.5, 0.8]] * int(T / dt)),
        ]).to(device)
    elif dphys_cfg.robot in ['tradr', 'tradr2']:
        # left, right
        controls = torch.stack([
            torch.tensor([[0.8, 0.5]] * int(T / dt)),
        ]).to(device)
    else:
        raise ValueError(f'Unknown robot: {dphys_cfg.robot}')
    B, N_ts = controls.shape[:2]
    assert controls.shape == (B, N_ts, len(dphys_cfg.driving_parts))

    # initial state
    x = torch.tensor([[0.0, 0.0, 0.2]], device=device).repeat(B, 1)
    assert x.shape == (B, 3)
    xd = torch.zeros_like(x)
    assert xd.shape == (B, 3)
    rs = torch.tensor(Rotation.from_euler('z', [0.0]).as_matrix(), dtype=torch.float32, device=device)
    assert rs.shape == (B, 3, 3)
    omega = torch.zeros_like(x)
    assert omega.shape == (B, 3)
    x_points = torch.as_tensor(dphys_cfg.robot_points, device=device).repeat(x.shape[0], 1, 1)
    assert x_points.shape == (B, len(dphys_cfg.robot_points), 3)
    x_points = x_points @ rs.transpose(1, 2) + x.unsqueeze(1)
    assert x_points.shape == (B, len(dphys_cfg.robot_points), 3)
    state0 = (x, xd, rs, omega, x_points)

    # heightmap defining the terrain
    x_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    y_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
    # z_grid = torch.sin(x_grid) * torch.cos(y_grid)
    z_grid = torch.exp(-(x_grid - 2) ** 2 / 4) * torch.exp(-(y_grid - 0) ** 2 / 2)
    # z_grid = torch.zeros_like(x_grid)
    x_grid, y_grid, z_grid = x_grid.to(device), y_grid.to(device), z_grid.to(device)
    stiffness = dphys_cfg.k_stiffness * torch.ones_like(z_grid)
    friction = dphys_cfg.k_friction * torch.ones_like(z_grid)
    # repeat the heightmap for each rigid body
    x_grid = x_grid.repeat(x.shape[0], 1, 1)
    y_grid = y_grid.repeat(x.shape[0], 1, 1)
    z_grid = z_grid.repeat(x.shape[0], 1, 1)
    stiffness = stiffness.repeat(x.shape[0], 1, 1)
    friction = friction.repeat(x.shape[0], 1, 1)
    H, W = int(2 * dphys_cfg.d_max / dphys_cfg.grid_res), int(2 * dphys_cfg.d_max / dphys_cfg.grid_res)
    assert x_grid.shape == (B, H, W)
    assert y_grid.shape == (B, H, W)
    assert z_grid.shape == (B, H, W)
    assert stiffness.shape == (B, H, W)
    assert friction.shape == (B, H, W)

    # simulate the rigid body dynamics
    states, forces = dphysics(z_grid=z_grid, controls=controls, state=state0,
                              stiffness=stiffness, friction=friction)

    # visualize using mayavi
    for b in range(len(states[0])):
        # get the states and forces for the b-th rigid body and move them to the cpu
        xs, xds, rs, omegas, x_points = [s[b].detach().cpu().numpy() for s in states]
        F_spring, F_friction = [f[b].detach().cpu().numpy() for f in forces]
        x_grid_np, y_grid_np, z_grid_np = [g[b].detach().cpu().numpy() for g in [x_grid, y_grid, z_grid]]
        friction_np = friction[b].detach().cpu().numpy()

        # set up the visualization
        vis_cfg = setup_visualization(states=(xs, xds, rs, omegas, x_points),
                                      forces=(F_spring, F_friction),
                                      x_grid=x_grid_np, y_grid=y_grid_np, z_grid=z_grid_np)

        # visualize animated trajectory
        animate_trajectory(states=(xs, xds, rs, omegas, x_points),
                           forces=(F_spring, F_friction),
                           z_grid=z_grid_np,
                           friction=friction_np,
                           vis_cfg=vis_cfg, step=10)


def motion_dataset():
    from monoforce.datasets import ROUGHBase, rough_seq_paths
    from monoforce.vis import set_axes_equal

    class Data(ROUGHBase):
        def __init__(self, path, dphys_cfg=DPhysConfig()):
            super(Data, self).__init__(path, dphys_cfg=dphys_cfg)

        def get_sample(self, i):
            _, controls = self.get_controls(i)
            _, states = self.get_states_traj(i)
            heightmap = self.get_geom_height_map(i)[0]
            return controls, states, heightmap

    # load the dataset
    path = np.random.choice(rough_seq_paths[robot])
    dphys_cfg = DPhysConfig(robot=robot)
    ds = Data(path, dphys_cfg=dphys_cfg)

    # instantiate the simulator
    dphysics = DPhysics(dphys_cfg, device=device)

    # get a sample from the dataset
    sample = ds[np.random.randint(len(ds))]
    controls, states, z_grid = sample

    # differentiable physics simulation
    z_grid = torch.as_tensor(z_grid)[None].to(device)
    controls = torch.as_tensor(controls)[None].to(device)
    states_pred, _ = dphysics(z_grid=z_grid, controls=controls)
    Xs_pred, Xds_pred, Rs_pred, Omegas_pred, _ = states_pred

    # helper quantities for visualization
    Xs = torch.as_tensor(states[0])[None]
    x_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    y_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)

    # visualize predicted and ground truth trajectories and the heightmap surface
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_grid.numpy(), y_grid.numpy(), z_grid[0].cpu().numpy(), alpha=0.6, cmap='terrain')
    ax.plot(Xs[0, :, 0].cpu().numpy(),
            Xs[0, :, 1].cpu().numpy(),
            Xs[0, :, 2].cpu().numpy(),
            c='b', label='GT trajectory')
    ax.plot(Xs_pred[0, :, 0].cpu().numpy(),
            Xs_pred[0, :, 1].cpu().numpy(),
            Xs_pred[0, :, 2].cpu().numpy(),
            c='r', label='Predicted trajectory')
    ax.set_title('Ground truth and predicted trajectories')
    ax.legend()
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    set_axes_equal(ax)
    plt.show()

def shoot_multiple():
    from time import time
    from scipy.spatial.transform import Rotation
    from monoforce.vis import set_axes_equal

    # simulation parameters
    dt = dphys_cfg.dt
    T = dphys_cfg.traj_sim_time
    num_trajs = dphys_cfg.n_sim_trajs
    vel_max, omega_max = dphys_cfg.vel_max, dphys_cfg.omega_max

    # instantiate the simulator
    dphysics = DPhysics(dphys_cfg, device=device)

    # rigid body parameters
    x_points = torch.as_tensor(dphys_cfg.robot_points, device=device)

    # initial state
    x = torch.tensor([[0.0, 0.0, 0.2]], device=device).repeat(num_trajs, 1)
    xd = torch.zeros_like(x)
    R = torch.eye(3, device=device).repeat(x.shape[0], 1, 1)
    # R = torch.tensor(Rotation.from_euler('z', np.pi/6).as_matrix(), dtype=torch.float32, device=device).repeat(num_trajs, 1, 1)
    omega = torch.zeros_like(x)
    x_points = x_points @ R.transpose(1, 2) + x.unsqueeze(1)

    # terrain properties
    x_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res).to(device)
    y_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res).to(device)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
    z_grid = torch.exp(-(x_grid - 2) ** 2 / 4) * torch.exp(-(y_grid - 0) ** 2 / 2)
    # z_grid = torch.sin(x_grid) * torch.cos(y_grid)
    # z_grid = torch.zeros_like(x_grid)

    stiffness = dphys_cfg.k_stiffness * torch.ones_like(z_grid)
    friction = dphys_cfg.k_friction * torch.ones_like(z_grid)
    # repeat the heightmap for each rigid body
    x_grid = x_grid.repeat(x.shape[0], 1, 1)
    y_grid = y_grid.repeat(x.shape[0], 1, 1)
    z_grid = z_grid.repeat(x.shape[0], 1, 1)
    stiffness = stiffness.repeat(x.shape[0], 1, 1)
    friction = friction.repeat(x.shape[0], 1, 1)

    # control inputs in m/s and rad/s
    controls_front, _ = generate_control_inputs(n_trajs=num_trajs // 2,
                                                robot_base=dphys_cfg.robot_size[1].item(),
                                                v_range=(vel_max / 2, vel_max), w_range=(-omega_max, omega_max),
                                                time_horizon=T, dt=dt)
    controls_back, _ = generate_control_inputs(n_trajs=num_trajs // 2,
                                               robot_base=dphys_cfg.robot_size[1].item(),
                                               v_range=(-vel_max, -vel_max / 2), w_range=(-omega_max, omega_max),
                                               time_horizon=T, dt=dt)
    controls = torch.cat([controls_front, controls_back], dim=0)
    assert num_trajs % 2 == 0, 'num_trajs must be even'
    controls = torch.as_tensor(controls, dtype=torch.float32, device=device)

    # initial state
    state0 = (x, xd, R, omega, x_points)

    # put tensors to device
    state0 = tuple([s.to(device) for s in state0])
    z_grid = z_grid.to(device)
    controls = controls.to(device)

    # simulate the rigid body dynamics
    with torch.no_grad():
        t0 = time()
        states, forces = dphysics(z_grid=z_grid, controls=controls, state=state0)
        t1 = time()
        Xs, Xds, Rs, Omegas, X_points = states
        print(f'Robot body points shape: {X_points.shape}')
        print(f'Simulation took {(t1-t0):.3f} [sec] on device: {device}')

    # visualize
    with torch.no_grad():
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # plot heightmap
        ax.plot_surface(x_grid[0].cpu().numpy(), y_grid[0].cpu().numpy(), z_grid[0].cpu().numpy(), alpha=0.6, cmap='terrain')
        set_axes_equal(ax)
        for i in range(num_trajs):
            ax.plot(Xs[i, :, 0].cpu(), Xs[i, :, 1].cpu(), Xs[i, :, 2].cpu(), c='b')
        ax.set_title(f'Simulation of {num_trajs} trajs (T={T} [sec] long) took {(t1-t0):.3f} [sec] on device: {device}')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        plt.show()


def main():
    motion()
    motion_dataset()
    shoot_multiple()


if __name__ == '__main__':
    main()
