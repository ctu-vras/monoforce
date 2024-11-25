#!/usr/bin/env python

import sys
sys.path.append('../src')
import torch
from monoforce.dphys_config import DPhysConfig
from monoforce.models.dphysics import DPhysics, generate_control_inputs
from monoforce.vis import setup_visualization, animate_trajectory
import matplotlib.pyplot as plt

# simulation parameters
robot = 'tradr'
dphys_cfg = DPhysConfig(robot=robot)
dphys_cfg.k_friction = 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def motion():
    from scipy.spatial.transform import Rotation

    # instantiate the simulator
    dphysics = DPhysics(dphys_cfg, device=device)

    # control inputs: linear velocity and angular velocity, v in m/s, w in rad/s
    controls = torch.stack([
        torch.tensor([[0.0, 0.0]] * int(dphys_cfg.traj_sim_time / dphys_cfg.dt)),  # [v] m/s, [w] rad/s for each time step
    ]).to(device)
    B, N_ts, _ = controls.shape
    assert controls.shape == (B, N_ts, 2), f'controls shape: {controls.shape}'

    # initial state
    x = torch.tensor([[0.5, 0.0, 1.0]], device=device).repeat(B, 1)
    assert x.shape == (B, 3)
    xd = torch.zeros_like(x)
    assert xd.shape == (B, 3)
    rs = torch.tensor(Rotation.from_euler('z', [0.0]).as_matrix(), dtype=torch.float32, device=device)
    assert rs.shape == (B, 3, 3)
    omega = torch.zeros_like(x)
    assert omega.shape == (B, 3)
    state0 = (x, xd, rs, omega)

    # heightmap defining the terrain
    x_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    y_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')
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
    xs, xds, rs, omegas = [s[0].detach().cpu().numpy() for s in states]
    F_spring, F_friction = [f[0].detach().cpu().numpy() for f in forces]
    x_grid_np, y_grid_np, z_grid_np = [g[0].detach().cpu().numpy() for g in [x_grid, y_grid, z_grid]]
    friction_np = friction[0].detach().cpu().numpy()
    x_points = dphys_cfg.robot_points.cpu().numpy()

    # set up the visualization
    vis_cfg = setup_visualization(states=(xs, xds, rs, omegas),
                                  x_points=x_points,
                                  forces=(F_spring, F_friction),
                                  x_grid=x_grid_np, y_grid=y_grid_np, z_grid=z_grid_np)

    # visualize animated trajectory
    animate_trajectory(states=(xs, xds, rs, omegas),
                       x_points=x_points,
                       forces=(F_spring, F_friction),
                       z_grid=z_grid_np,
                       friction=friction_np,
                       vis_cfg=vis_cfg, step=10)


def motion_dataset():
    import numpy as np
    from monoforce.datasets import ROUGH, rough_seq_paths
    from monoforce.utils import explore_data
    from scipy.spatial.transform import Rotation

    # load the dataset
    path = rough_seq_paths[0]
    # path = np.random.choice(rough_seq_paths)
    ds = ROUGH(path, dphys_cfg=dphys_cfg)

    # instantiate the simulator
    dphysics = DPhysics(dphys_cfg, device=device)

    # helper quantities for visualization
    x_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    y_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')

    sample_i = 120
    # sample_i = np.random.choice(len(ds))
    print(f'Sample index: {sample_i}')
    # get a sample from the dataset
    traj_ts, states = ds.get_states_traj(sample_i)
    hm = ds.get_terrain_height_map(sample_i)
    control_ts, controls = ds.get_controls(sample_i)
    z_grid, grid_mask = hm[0], hm[1]

    # interpolate the heightmap with minimum measured height
    z_grid[~grid_mask.bool()] = z_grid[grid_mask.bool()].min()

    # initial state
    pose0 = torch.as_tensor(ds.get_initial_pose_on_heightmap(sample_i), dtype=torch.float32)
    x = pose0[:3, 3]
    xd = torch.zeros_like(x)
    R = pose0[:3, :3]
    omega = torch.zeros_like(x)
    state0 = (x, xd, R, omega)
    state0 = tuple([s.to(device)[None] for s in state0])

    explore_data(ds, sample_range=[sample_i])
    # # plot controls
    # plt.figure()
    # plt.plot(controls[:, 0], '.', label='v [m/s]')
    # plt.plot(controls[:, 1], '.', label='w [rad/s]')
    # plt.legend()
    # plt.title('Controls')
    # plt.show()

    # differentiable physics simulation
    states_pred, forces_pred = dphysics(z_grid=torch.as_tensor(z_grid)[None].to(device),
                                        state=state0,
                                        controls=torch.as_tensor(controls)[None].to(device))

    # get the states and forces for and move them to the cpu
    states_pred_np = [s.squeeze(0).cpu().numpy() for s in states_pred]
    forces_pred_np = [f.squeeze(0).cpu().numpy() for f in forces_pred]
    x_points = dphys_cfg.robot_points.cpu().numpy()

    # set up the visualization
    vis_cfg = setup_visualization(states=states_pred_np,
                                  x_points=x_points,
                                  states_gt=states,
                                  forces=forces_pred_np,
                                  x_grid=x_grid.cpu().numpy(), y_grid=y_grid.cpu().numpy(), z_grid=z_grid.cpu().numpy())

    # visualize animated trajectory
    animate_trajectory(states=states_pred_np,
                       x_points=x_points,
                       forces=forces_pred_np,
                       z_grid=z_grid.cpu().numpy(),
                       vis_cfg=vis_cfg, step=10)

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

    # initial state
    x = torch.tensor([[0.0, 0.0, 0.0]], device=device).repeat(num_trajs, 1)
    xd = torch.zeros_like(x)
    R = torch.eye(3, device=device).repeat(x.shape[0], 1, 1)
    # R = torch.tensor(Rotation.from_euler('z', np.pi/6).as_matrix(), dtype=torch.float32, device=device).repeat(num_trajs, 1, 1)
    omega = torch.zeros_like(x)

    # terrain properties
    x_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res).to(device)
    y_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res).to(device)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')
    z_grid = torch.exp(-(x_grid - 2) ** 2 / 4) * torch.exp(-(y_grid - 0) ** 2 / 2)
    # z_grid = torch.sin(x_grid) * torch.cos(y_grid)
    # z_grid = torch.zeros_like(x_grid)

    # repeat the heightmap for each rigid body
    x_grid = x_grid.repeat(num_trajs, 1, 1)
    y_grid = y_grid.repeat(num_trajs, 1, 1)
    z_grid = z_grid.repeat(num_trajs, 1, 1)

    # control inputs in m/s and rad/s
    controls_front, _ = generate_control_inputs(n_trajs=num_trajs // 2,
                                                v_range=(vel_max / 2, vel_max), w_range=(-omega_max, omega_max),
                                                time_horizon=T, dt=dt)
    controls_back, _ = generate_control_inputs(n_trajs=num_trajs // 2,
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
        states, forces = dphysics(z_grid=z_grid, controls=controls, state=state0)
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
    motion_dataset()
    shoot_multiple()


if __name__ == '__main__':
    main()
