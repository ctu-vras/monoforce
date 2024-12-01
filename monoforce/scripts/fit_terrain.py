import sys

from IPython.core.pylabtools import figsize

sys.path.append('../src')
import numpy as np
import torch
from monoforce.models.dphysics import DPhysics
from monoforce.dphys_config import DPhysConfig
from monoforce.vis import setup_visualization, animate_trajectory
from monoforce.datasets import ROUGH, rough_seq_paths
from time import time
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')


def visualize(states, x_points, forces, x_grid, y_grid, z_grid, states_gt=None):
    with torch.no_grad():
        # visualize using mayavi
        for b in [0]:
            # get the states and forces for the b-th rigid body and move them to the cpu
            xs, xds, Rs, omegas = [s[b].cpu().numpy() for s in states]
            F_spring, F_friction = [f[b].cpu().numpy() for f in forces]
            x_grid_np, y_grid_np, z_grid_np = [g[b].cpu().numpy() for g in [x_grid, y_grid, z_grid]]

            # set up the visualization
            vis_cfg = setup_visualization(states=(xs, xds, Rs, omegas),
                                          x_points=x_points,
                                          states_gt=[s[b].cpu().numpy() for s in states_gt] if states_gt else None,
                                          forces=(F_spring, F_friction),
                                          x_grid=x_grid_np, y_grid=y_grid_np, z_grid=z_grid_np)

            # visualize animated trajectory
            animate_trajectory(states=(xs, xds, Rs, omegas),
                               x_points=x_points,
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
    lr = 0.02
    vis = False

    # initial state
    x = torch.tensor([[-1.0, 0.0, 0.1]])
    xd = torch.tensor([[0.0, 0.0, 0.0]])
    R = torch.eye(3).repeat(x.shape[0], 1, 1)
    omega = torch.tensor([[0.0, 0.0, 0.0]])

    # heightmap defining the terrain
    x_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    y_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')
    z_grid_gt = torch.exp(-(x_grid - 2) ** 2 / 4) * torch.exp(-(y_grid - 0) ** 2 / 2)
    # repeat the heightmap for each rigid body
    x_grid = x_grid.repeat(x.shape[0], 1, 1)
    y_grid = y_grid.repeat(x.shape[0], 1, 1)
    z_grid_gt = z_grid_gt.repeat(x.shape[0], 1, 1)

    # control inputs in m/s
    controls = torch.tensor([[[1.0, 1.0]] * int(dphys_cfg.traj_sim_time / dphys_cfg.dt)])

    # initial state
    state0 = (x, xd, R, omega)
    x_points = dphys_cfg.robot_points.cpu().numpy()

    # simulate the rigid body dynamics
    dphysics = DPhysics(dphys_cfg)
    states_gt, forces_gt = dphysics(z_grid=z_grid_gt, controls=controls, state=state0)
    if vis:
        visualize(states_gt, x_points, forces_gt, x_grid, y_grid, z_grid_gt, states_gt)

    # initial guess for the heightmap
    z_grid = torch.zeros_like(z_grid_gt, requires_grad=True)

    # optimization: height and friction with different learning rates
    optimizer = torch.optim.Adam([z_grid], lr=lr)
    Xs_gt, Xds_gt, Rs_gt, Omegas_gt = states_gt
    loss_min = np.Inf
    z_grid_best = z_grid.clone()
    state_dT = None
    for i in range(n_iters):
        optimizer.zero_grad()
        loss = torch.tensor(0.0, requires_grad=True)
        for t in range(0, int(T / dt), int(dT / dt)):
            # simulate the rigid body dynamics for dT time period
            dphys_cfg.traj_sim_time = dT
            # state_gt = (Xs_gt[:, t], Xds_gt[:, t], Rs_gt[:, t], Omegas_gt[:, t])
            state_dT = state0 if state_dT is None else [s[:, -1] for s in states_dT]
            controls_dT = controls[:, t: t + int(dT / dt)]
            states_dT, forces_dT = dphysics(z_grid=z_grid, controls=controls_dT,
                                            # state=state0 if t == 0 else state_gt,
                                            state=state0 if t == 0 else state_dT)
            # unpack the states
            Xs, Xds, Rs, Omegas = states_dT

            # compute the loss
            loss_dT = torch.nn.functional.mse_loss(Xs, Xs_gt[:, t:t + int(dT / dt)])
            loss = loss + loss_dT

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
            visualize(states, x_points, forces, x_grid, y_grid, z_grid, states_gt)

    # visualize the best heightmap
    dphys_cfg.traj_sim_time = T
    states, forces = dphysics(z_grid=z_grid_best, controls=controls, state=state0)
    visualize(states, x_points, forces, x_grid, y_grid, z_grid_best, states_gt)


def learn_terrain_properties():
    vis = True
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
    path = rough_seq_paths[0]

    # load the dataset
    ds = Data(path, dphys_cfg=dphys_cfg)

    # instantiate the simulator
    dphysics = DPhysics(dphys_cfg, device=device)

    # get a sample from the dataset
    # sample_i = np.random.choice(len(ds))
    sample_i = 79
    print('Sample index:', sample_i)
    sample = ds[sample_i]
    batch = [s[None].to(device) for s in sample]
    control_ts, controls, traj_ts, X, Xd, R, Omega, z_grid = batch
    batch_size = X.shape[0]

    z_grid = torch.zeros_like(z_grid)
    friction = 0.0 * torch.ones_like(z_grid)

    # make gt traj sparser
    traj_ts = traj_ts[:, ::10]
    X = X[:, ::10]

    # find the closest timesteps in the trajectory to the ground truth timesteps
    t0 = time()
    ts_ids = torch.argmin(torch.abs(control_ts.unsqueeze(1) - traj_ts.unsqueeze(2)), dim=2)
    print(f'Finding closest timesteps took: {time() - t0} [sec]')

    # optimize the heightmap and friction
    z_grid.requires_grad = True
    friction.requires_grad = True
    optimizer = torch.optim.Adam([{'params': z_grid, 'lr': 0.0},
                                  {'params': friction, 'lr': 0.05}])

    print('Optimizing terrain properties...')
    n_iters, vis_step = 100, 10
    plt.figure(figsize(10, 5))
    for i in range(n_iters):
        optimizer.zero_grad()
        states_pred, forces_pred = dphysics(z_grid=z_grid, controls=controls, friction=friction)
        X_pred, Xd_pred, R_pred, Omega_pred = states_pred

        # compute the loss as the mean squared error between the predicted and ground truth poses
        loss = torch.nn.functional.mse_loss(X_pred[torch.arange(batch_size).unsqueeze(1), ts_ids], X)
        loss.backward()
        optimizer.step()
        print(f'Iteration {i}, Loss: {loss.item()}')
        # print(f'Friction mean: {friction.mean().item()}')

        if vis and i % vis_step == 0:
            with torch.no_grad():
                plt.clf()

                plt.subplot(121)
                plt.title('Trajectory Z(t)')
                plt.plot(traj_ts[0].cpu().numpy(), X[0, :, 2].cpu().numpy(), '.b')
                plt.plot(control_ts[0].cpu().numpy(), X_pred[0, :, 2].cpu().numpy(), '.r')
                plt.xlabel('Time [s]')
                plt.ylabel('Z [m]')
                plt.grid()
                plt.ylim(-1, 1)
                plt.xlim(-0.1, 5.1)

                plt.subplot(122)
                plt.title('Trajectory Y(X)')
                plt.plot(X[0, :, 0].cpu().numpy(), X[0, :, 1].cpu().numpy(), '.b')
                plt.plot(X_pred[0, :, 0].cpu().numpy(), X_pred[0, :, 1].cpu().numpy(), '.r')
                # plot lines between corresponding points from the ground truth and predicted trajectories (use ts_ids)
                for j in range(X.shape[1]):
                    plt.plot([X[0, j, 0].cpu().numpy(), X_pred[0, ts_ids[0, j], 0].cpu().numpy()],
                             [X[0, j, 1].cpu().numpy(), X_pred[0, ts_ids[0, j], 1].cpu().numpy()], 'g')
                plt.xlabel('X [m]')
                plt.ylabel('Y [m]')
                plt.grid()
                plt.ylim(-6.4, 6.4)
                plt.xlim(-6.4, 6.4)

                plt.pause(0.1)
                plt.draw()

                # x_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
                # y_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
                # x_grid, y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')
                # x_grid = x_grid.repeat(batch_size, 1, 1)
                # y_grid = y_grid.repeat(batch_size, 1, 1)
                # x_points = dphys_cfg.robot_points.cpu().numpy()
                # states = tuple(batch[3:7])
                # visualize(states=states_pred,
                #           x_points=x_points,
                #           states_gt=states,
                #           forces=forces_pred,
                #           x_grid=x_grid, y_grid=y_grid, z_grid=z_grid)


def main():
    # optimize_heightmap()
    learn_terrain_properties()


if __name__ == '__main__':
    main()
