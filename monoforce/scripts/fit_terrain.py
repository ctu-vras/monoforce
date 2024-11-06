import sys
sys.path.append('../src')
import numpy as np
import torch
from monoforce.models.dphysics import DPhysics
from monoforce.dphys_config import DPhysConfig
from monoforce.vis import setup_visualization, animate_trajectory


def visualize(states, forces, x_grid, y_grid, z_grid):
    with torch.no_grad():
        # visualize using mayavi
        for b in range(len(states[0])):
            # get the states and forces for the b-th rigid body and move them to the cpu
            xs, R, xds, omegas, x_points = [s[b].cpu().numpy() for s in states]
            F_spring, F_friction = [f[b].cpu().numpy() for f in forces]
            x_grid_np, y_grid_np, z_grid_np = [g[b].cpu().numpy() for g in [x_grid, y_grid, z_grid]]

            # set up the visualization
            vis_cfg = setup_visualization(states=(xs, R, xds, omegas, x_points),
                                          forces=(F_spring, F_friction),
                                          x_grid=x_grid_np, y_grid=y_grid_np, z_grid=z_grid_np)

            # visualize animated trajectory
            animate_trajectory(states=(xs, R, xds, omegas, x_points),
                               forces=(F_spring, F_friction),
                               z_grid=z_grid_np,
                               vis_cfg=vis_cfg, step=10)


def optimize_heightmap():
    dphys_cfg = DPhysConfig()

    # simulation parameters
    T, dt = 3.0, 0.01
    dT = 0.2  # time period to cut the trajectory into samples
    dphys_cfg.dt = dt
    dphys_cfg.traj_sim_time = T
    n_iters = 100
    lr = 0.01
    vis = True

    # rigid body parameters
    x_points = torch.as_tensor(dphys_cfg.robot_points)

    # initial state
    x = torch.tensor([[-1.0, 0.0, 0.1]])
    xd = torch.tensor([[0.0, 0.0, 0.0]])
    R = torch.eye(3).repeat(x.shape[0], 1, 1)
    omega = torch.tensor([[0.0, 0.0, 0.0]])
    x_points = x_points @ R.transpose(1, 2) + x.unsqueeze(1)

    # heightmap defining the terrain
    x_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    y_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
    z_grid_gt = torch.exp(-(x_grid - 2) ** 2 / 4) * torch.exp(-(y_grid - 0) ** 2 / 2)
    # repeat the heightmap for each rigid body
    x_grid = x_grid.repeat(x.shape[0], 1, 1)
    y_grid = y_grid.repeat(x.shape[0], 1, 1)
    z_grid_gt = z_grid_gt.repeat(x.shape[0], 1, 1)

    # control inputs in m/s
    controls = torch.tensor([[[1.0, 1.0]] * int(dphys_cfg.traj_sim_time / dphys_cfg.dt)])

    # initial state
    state0 = (x, xd, R, omega, x_points)

    # simulate the rigid body dynamics
    dphysics = DPhysics(dphys_cfg)
    states_gt, forces_gt = dphysics(z_grid=z_grid_gt, controls=controls, state=state0)
    if vis:
        visualize(states_gt, forces_gt, x_grid, y_grid, z_grid_gt)

    # initial guess for the heightmap
    z_grid = torch.zeros_like(z_grid_gt, requires_grad=True)

    # optimization: height and friction with different learning rates
    optimizer = torch.optim.Adam([z_grid], lr=lr)
    Xs_gt, Xds_gt, Rs_gt, Omegas_gt, X_points_gt = states_gt
    loss_min = np.Inf
    z_grid_best = z_grid.clone()
    state_dT = None
    for i in range(n_iters):
        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        for t in range(0, int(T / dt), int(dT / dt)):
            # simulate the rigid body dynamics for dT time period
            dphys_cfg.traj_sim_time = dT
            # state_gt = (Xs_gt[:, t], Xds_gt[:, t], Rs_gt[:, t], Omegas_gt[:, t], X_points_gt[:, t])
            state_dT = state0 if state_dT is None else [s[:, -1] for s in states_dT]
            controls_dT = controls[:, t: t + int(dT / dt)]
            states_dT, forces_dT = dphysics(z_grid=z_grid, controls=controls_dT,
                                            # state=state0 if t == 0 else state_gt,
                                            state=state0 if t == 0 else state_dT)
            # unpack the states
            Xs, Xds, Rs, Omegas, X_points = states_dT

            # compute the loss
            loss_dT = torch.nn.functional.mse_loss(Xs, Xs_gt[:, t:t + int(dT / dt)])
            loss += loss_dT

        loss.backward()
        optimizer.step()
        print(f'Iteration {i}, Loss_x: {loss.item()}')

        if loss.item() < loss_min:
            loss_min = loss.item()
            z_grid_best = z_grid.clone()

        # heightmap difference
        with torch.no_grad():
            z_diff = torch.nn.functional.mse_loss(z_grid, z_grid_gt)
            print(f'Heightmap difference: {z_diff.item()}')

        # visualize the optimized heightmap
        if vis and i % 20 == 0:
            dphys_cfg.traj_sim_time = T
            states, forces = dphysics(z_grid=z_grid, controls=controls, state=state0)
            visualize(states, forces, x_grid, y_grid, z_grid)

    # visualize the best heightmap
    dphys_cfg.traj_sim_time = T
    states, forces = dphysics(z_grid=z_grid_best, controls=controls, state=state0)
    visualize(states, forces, x_grid, y_grid, z_grid_best)


def learn_terrain_properties():
    import matplotlib.pyplot as plt
    from monoforce.datasets import ROUGH, rough_seq_paths
    from torch.utils.data import DataLoader
    from time import time

    np.random.seed(0)
    torch.manual_seed(0)
    vis = True
    batch_size = 4
    device = torch.device('cuda')

    class Data(ROUGH):
        def __init__(self, path, dphys_cfg=DPhysConfig()):
            super(Data, self).__init__(path, dphys_cfg=dphys_cfg)

        def get_sample(self, i):
            control_ts, controls = self.get_controls(i)
            ts, states = self.get_states_traj(i)
            Xs, Xds, Rs, Omegas = states
            heightmap = self.get_geom_height_map(i)[0]
            return control_ts, controls, ts, Xs, Xds, Rs, Omegas, heightmap

    dphys_cfg = DPhysConfig()
    path = rough_seq_paths['tradr'][0]

    # load the dataset
    ds = Data(path, dphys_cfg=dphys_cfg)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # instantiate the simulator
    dphysics = DPhysics(dphys_cfg, device=device)

    # get a sample from the dataset
    control_ts, controls, traj_ts, Xs, Xds, Rs, Omegas, z_grid = next(iter(loader))
    states = [Xs, Xds, Rs, Omegas]
    friction = dphys_cfg.k_friction * torch.ones_like(z_grid)

    # put the data on the device
    control_ts, controls, traj_ts = [t.to(device) for t in [control_ts, controls, traj_ts]]
    states = [s.to(device) for s in states]
    z_grid = z_grid.to(device)
    friction = friction.to(device)

    # find the closest timesteps in the trajectory to the ground truth timesteps
    t0 = time()
    ts_ids = torch.argmin(torch.abs(control_ts.unsqueeze(1) - traj_ts.unsqueeze(2)), dim=2)
    print(f'Finding closest timesteps took: {time() - t0} [sec]')

    # optimize the heightmap and friction
    z_grid.requires_grad = True
    friction.requires_grad = True
    optimizer = torch.optim.Adam([{'params': z_grid, 'lr': 0.001},
                                  {'params': friction, 'lr': 0.02}])

    print('Optimizing terrain properties...')
    n_iters, vis_step = 100, 10
    for i in range(n_iters):
        optimizer.zero_grad()
        states_pred, _ = dphysics(z_grid=z_grid, controls=controls, friction=friction)

        X, Xd, R, Omega = states
        X_pred, Xd_pred, R_pred, Omega_pred, _ = states_pred

        # compute the loss as the mean squared error between the predicted and ground truth poses
        loss = torch.nn.functional.mse_loss(X_pred[torch.arange(batch_size).unsqueeze(1), ts_ids], X)
        loss.backward()
        optimizer.step()
        print(f'Iteration {i}, Loss: {loss.item()}')

        if vis and i % vis_step == 0:
            with torch.no_grad():
                # for batch_i in range(batch_size):
                for batch_i in [np.random.choice(batch_size)]:
                    plt.figure(figsize=(20, 10))
                    plt.subplot(121)
                    plt.title(f'Trajectories loss: {loss.item()}')
                    xyz_pred = states_pred[0].cpu().numpy()[batch_i]
                    xyz = states[0].cpu().numpy()[batch_i]
                    ids = ts_ids[batch_i].cpu().numpy()
                    xyz_pred_grid = (xyz_pred + dphys_cfg.d_max) / dphys_cfg.grid_res
                    xyz_grid = (xyz + dphys_cfg.d_max) / dphys_cfg.grid_res
                    plt.plot(xyz_grid[:, 0], xyz_grid[:, 1], 'kx', label='Ground truth')
                    plt.plot(xyz_pred_grid[ids, 0], xyz_pred_grid[ids, 1], 'rx', label='Predicted')
                    # plt.imshow(z_grid[0].cpu().numpy().T, origin='lower')
                    # plt.xlim([0, z_grid.shape[-1]])
                    # plt.ylim([0, z_grid.shape[-2]])
                    plt.axis('equal')
                    plt.legend()
                    plt.grid()

                    plt.subplot(122)
                    plt.title(f'Mean friction: {friction.mean().item()}')
                    plt.plot(xyz_pred_grid[ids, 0], xyz_pred_grid[ids, 1], 'r.')
                    plt.imshow(friction[0].cpu().numpy().T, origin='lower')
                    plt.colorbar()
                    plt.grid()

                    plt.show()


def main():
    # optimize_heightmap()
    learn_terrain_properties()


if __name__ == '__main__':
    main()
