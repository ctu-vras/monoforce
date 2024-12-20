import sys
sys.path.append('../src')
import numpy as np
import torch
from monoforce.models.dphysics import DPhysics
from monoforce.dphys_config import DPhysConfig
from monoforce.vis import setup_visualization, animate_trajectory
from monoforce.datasets import ROUGH, rough_seq_paths
from monoforce.losses import physics_loss, hm_loss, total_variation
import os
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
            x_points = x_points.cpu().numpy()

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


def optimize_terrain():
    dphys_cfg = DPhysConfig()

    # simulation parameters
    T, dt = 2.0, 0.01
    dphys_cfg.dt = dt
    dphys_cfg.traj_sim_time = T
    n_iters = 100
    vis = True

    # initial state
    x = torch.tensor([[-1.0, 0.0, 0.1]])
    xd = torch.tensor([[0.0, 0.0, 0.0]])
    R = torch.eye(3).repeat(x.shape[0], 1, 1)
    omega = torch.tensor([[0.0, 0.0, 0.0]])

    # heightmap defining the terrain
    x_grid, y_grid = dphys_cfg.x_grid, dphys_cfg.y_grid
    z_grid_gt = torch.exp(-(x_grid - 2) ** 2 / 4) * torch.exp(-(y_grid - 0) ** 2 / 2)
    # repeat the heightmap for each rigid body
    x_grid = x_grid.repeat(x.shape[0], 1, 1)
    y_grid = y_grid.repeat(x.shape[0], 1, 1)
    z_grid_gt = z_grid_gt.repeat(x.shape[0], 1, 1)
    friction_gt = dphys_cfg.friction.repeat(x.shape[0], 1, 1)

    # control inputs in m/s
    controls = torch.tensor([[[1.0, 1.0]] * int(dphys_cfg.traj_sim_time / dphys_cfg.dt)])

    # initial state
    state0 = (x, xd, R, omega)
    x_points = dphys_cfg.robot_points

    # simulate the rigid body dynamics
    dphysics = DPhysics(dphys_cfg)
    states_gt, forces_gt = dphysics(z_grid=z_grid_gt, controls=controls, state=state0, friction=friction_gt)
    if vis:
        visualize(states_gt, x_points, forces_gt, x_grid, y_grid, z_grid_gt, states_gt)

    # initial guess for the heightmap
    z_grid = torch.zeros_like(z_grid_gt, requires_grad=True)
    friction = 1.0 * torch.ones_like(friction_gt)
    friction.requires_grad = True

    # optimization: height and friction with different learning rates
    optimizer = torch.optim.Adam([{'params': z_grid, 'lr': 0.02}, {'params': friction, 'lr': 0.0}])

    loss_min = np.inf
    z_grid_best = z_grid.clone()
    friction_best = friction.clone()
    losses_history = {'traj': [], 'hdiff': [], 'total': []}
    ts = torch.arange(0, T, dt)[None]
    for i in range(n_iters):
        optimizer.zero_grad()

        states, _ = dphysics(z_grid=z_grid, controls=controls, state=state0, friction=friction)
        loss_traj = physics_loss(states_pred=states, states_gt=states_gt, pred_ts=ts, gt_ts=ts, gamma=0.)
        loss_terrain = total_variation(z_grid)
        loss = loss_traj + loss_terrain

        loss.backward()
        optimizer.step()
        print(f'Iteration {i}, Loss: {loss.item()}')
        # print(f'Friction mean: {friction.mean().item()}')
        # print(f'Heightmap mean: {z_grid.mean().item()}')

        if loss.item() < loss_min:
            loss_min = loss.item()
            z_grid_best = z_grid.clone()
            friction_best = friction.clone()

        # visualize the optimized heightmap
        if vis and i % 20 == 0:
            states, forces = dphysics(z_grid=z_grid, controls=controls, state=state0, friction=friction)
            visualize(states, x_points, forces, x_grid, y_grid, z_grid, states_gt)

        losses_history['traj'].append(loss_traj.item())
        losses_history['hdiff'].append(loss_terrain.item())
        losses_history['total'].append(loss.item())

    plt.figure()
    for key, loss in losses_history.items():
        plt.plot(loss, label=key)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    # visualize the best heightmap
    states, forces = dphysics(z_grid=z_grid_best, controls=controls, state=state0, friction=friction_best)
    visualize(states, x_points, forces, x_grid, y_grid, z_grid_best, states_gt)


def optimize_terrain_heads():
    from monoforce.models.terrain_encoder.lss import LiftSplatShoot
    from monoforce.utils import read_yaml, explore_data

    n_iters = 40
    vis_step = 10
    vis = True
    device = torch.device('cuda')

    dphys_cfg = DPhysConfig()
    dphys_cfg.traj_sim_time = 2.0
    path = rough_seq_paths[2]
    lss_cfg = read_yaml('../config/lss_cfg.yaml')

    # load the dataset
    ds = ROUGH(path, dphys_cfg=dphys_cfg, lss_cfg=lss_cfg)

    lss = LiftSplatShoot(grid_conf=lss_cfg['grid_conf'], data_aug_conf=lss_cfg['data_aug_conf'])
    lss.to(device)

    # instantiate the simulator
    dphysics = DPhysics(dphys_cfg, device=device)

    # get a sample from the dataset
    # sample_i = 270
    sample_i = np.random.choice(len(ds))
    print(f'Sample index: {sample_i}')
    explore_data(ds, sample_range=[sample_i])
    batch = [s[None] for s in ds[sample_i]]
    # loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    # batch = next(iter(loader))

    batch = [b.to(device) for b in batch]
    (imgs, rots, trans, intrins, post_rots, post_trans,
     hm_geom, hm_terrain,
     control_ts, controls,
     pose0,
     traj_ts, X, Xd, R, Omega) = batch

    # Freeze all layers
    for param in lss.parameters():
        param.requires_grad = False
    # Unfreeze the up_friction layer
    for param in lss.bevencode.up_friction.parameters():
        param.requires_grad = True
    for param in lss.bevencode.up_geom.parameters():
        param.requires_grad = True

    # optimize the model parameters
    lss.eval()
    lss.bevencode.up_friction.train()
    lss.bevencode.up_geom.train()
    optimizer = torch.optim.Adam([{'params': lss.bevencode.up_friction.parameters(), 'lr': 1e-3},
                                  {'params': lss.bevencode.up_geom.parameters(), 'lr': 1e-4}])
    # for name, param in lss.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    losses = []
    img_inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
    states_gt = [X, Xd, R, Omega]
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(n_iters):
        optimizer.zero_grad()

        terrain = lss(*img_inputs)
        z_grid = terrain['geom'].squeeze(1)
        friction = terrain['friction'].squeeze(1)
        states_pred, forces_pred = dphysics(z_grid=z_grid, controls=controls, friction=friction)
        X_pred, Xd_pred, R_pred, Omega_pred = states_pred

        # compute the loss as the mean squared error between the predicted and ground truth poses
        loss = physics_loss(states_pred=states_pred, states_gt=states_gt, pred_ts=control_ts, gt_ts=traj_ts,
                            gamma=1., rotation_loss=True)
        loss.backward()
        optimizer.step()
        print(f'Iteration {i}, Loss: {loss.item()}')
        losses.append(loss.item())

        if i % vis_step == 0 or i == n_iters - 1:
            with torch.no_grad():
                for axis in ax.flatten():
                    axis.clear()
                batch_i = np.random.choice(X.shape[0])
                ax[0, 0].set_title('Trajectory Z(t)')
                ax[0, 0].plot(traj_ts[batch_i].cpu().numpy(), X[batch_i, :, 2].cpu().numpy(), '.b')
                ax[0, 0].plot(control_ts[batch_i].cpu().numpy(), X_pred[batch_i, :, 2].cpu().numpy(), '.r')
                ax[0, 0].set_xlabel('Time [s]')
                ax[0, 0].set_ylabel('Z [m]')
                ax[0, 0].grid()
                ax[0, 0].set_ylim(-1, 1)
                ax[0, 0].set_xlim(-0.1, 5.1)

                ax[0, 1].set_title('Trajectory Y(X)')
                ax[0, 1].plot(X[batch_i, :, 0].cpu().numpy(), X[batch_i, :, 1].cpu().numpy(), '.b', label='GT')
                ax[0, 1].plot(X_pred[batch_i, :, 0].cpu().numpy(), X_pred[batch_i, :, 1].cpu().numpy(), '.r', label='Pred')
                # ts_ids = torch.argmin(torch.abs(control_ts.unsqueeze(1) - traj_ts.unsqueeze(2)), dim=2)
                # for j in range(X.shape[1]):
                #     ax[0, 1].plot([X[batch_i, j, 0].cpu().numpy(), X_pred[batch_i, ts_ids[batch_i, j], 0].cpu().numpy()],
                #                   [X[batch_i, j, 1].cpu().numpy(), X_pred[batch_i, ts_ids[batch_i, j], 1].cpu().numpy()], 'g')
                ax[0, 1].set_xlabel('X [m]')
                ax[0, 1].set_ylabel('Y [m]')
                ax[0, 1].grid()
                ax[0, 1].axis('equal')
                ax[0, 1].legend()

                ax[0, 2].set_title('Control inputs: V(t) and W(t)')
                ax[0, 2].plot(control_ts[batch_i].cpu().numpy(), controls[batch_i, :, 0].cpu().numpy(), '.b', label='V')
                ax[0, 2].plot(control_ts[batch_i].cpu().numpy(), controls[batch_i, :, 1].cpu().numpy(), '.r', label='W')
                ax[0, 2].set_xlabel('Time [s]')
                ax[0, 2].set_ylabel('Control inputs')
                ax[0, 2].grid()
                ax[0, 2].legend()

                ax[1, 0].set_title('Heightmap')
                ax[1, 0].imshow(z_grid[batch_i].cpu().numpy().T, cmap='jet', origin='lower', vmin=-1, vmax=1)
                ax[1, 0].set_xlabel('X')
                ax[1, 0].set_ylabel('Y')
                ax[1, 0].grid()
                x, y = X[batch_i, :, 0].cpu().numpy(), X[batch_i, :, 1].cpu().numpy()
                traj_x_grid = (x + dphys_cfg.d_max) / dphys_cfg.grid_res
                traj_y_grid = (y + dphys_cfg.d_max) / dphys_cfg.grid_res
                x_pred, y_pred = X_pred[batch_i, :, 0].cpu().numpy(), X_pred[batch_i, :, 1].cpu().numpy()
                traj_x_pred_grid = (x_pred + dphys_cfg.d_max) / dphys_cfg.grid_res
                traj_y_pred_grid = (y_pred + dphys_cfg.d_max) / dphys_cfg.grid_res
                ax[1, 0].plot(traj_x_grid, traj_y_grid, '.b', label='GT')
                ax[1, 0].plot(traj_x_pred_grid, traj_y_pred_grid, '.r', label='Pred')
                ax[1, 0].legend()

                ax[1, 1].set_title('Friction')
                ax[1, 1].imshow(friction[batch_i].cpu().numpy().T, cmap='jet', origin='lower', vmin=0, vmax=1)
                ax[1, 1].set_xlabel('X')
                ax[1, 1].set_ylabel('Y')
                ax[1, 1].grid()
                ax[1, 1].plot(traj_x_grid, traj_y_grid, '.b', label='GT')
                ax[1, 1].plot(traj_x_pred_grid, traj_y_pred_grid, '.r', label='Pred')
                ax[1, 1].legend()

                ax[1, 2].set_title('Loss')
                ax[1, 2].plot(losses, label='traj')
                ax[1, 2].set_xlabel('Iteration')
                ax[1, 2].set_ylabel('Loss')
                ax[1, 2].grid()
                ax[1, 2].legend()

                if vis:
                    plt.pause(0.01)
                    plt.draw()

                os.makedirs('./gen/terrain_optimization/', exist_ok=True)
                # 4 digits in file name
                plt.savefig(f'./gen/terrain_optimization/iter_{i:04d}.png')

    if vis:
        plt.show()
        visualize(states_pred, dphys_cfg.robot_points, forces_pred,
                  dphys_cfg.x_grid[None], dphys_cfg.y_grid[None], z_grid, [X])


def optimize_model():
    from monoforce.models.terrain_encoder.lss import LiftSplatShoot
    from monoforce.utils import read_yaml, explore_data

    vis = True
    device = torch.device('cuda')

    dphys_cfg = DPhysConfig()
    dphys_cfg.traj_sim_time = 2.0
    path = rough_seq_paths[0]
    lss_cfg = read_yaml('../config/lss_cfg.yaml')

    # load the dataset
    ds = ROUGH(path, dphys_cfg=dphys_cfg, lss_cfg=lss_cfg)

    # instantiate the model
    lss = LiftSplatShoot(grid_conf=lss_cfg['grid_conf'], data_aug_conf=lss_cfg['data_aug_conf'])
    lss.to(device)

    # instantiate the simulator
    dphysics = DPhysics(dphys_cfg, device=device)

    # get a sample from the dataset
    sample_i = 79
    explore_data(ds, sample_range=[sample_i])
    batch = [s[None] for s in ds[sample_i]]
    # loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    # batch = next(iter(loader))

    # put the batch on the device
    batch = [b.to(device) for b in batch]
    (imgs, rots, trans, intrins, post_rots, post_trans,
     hm_geom, hm_terrain,
     control_ts, controls,
     pose0,
     traj_ts, X, Xd, R, Omega) = batch

    # optimize the model parameters
    lss.train()
    optimizer = torch.optim.Adam(lss.parameters(), lr=1e-5)

    n_iters = 40
    losses = []

    img_inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
    states_gt = [X, Xd, R, Omega]
    for i in range(n_iters):
        optimizer.zero_grad()

        # forward pass
        terrain = lss(*img_inputs)
        z_grid = terrain['geom'].squeeze(1)
        friction = terrain['friction'].squeeze(1)
        states_pred, forces_pred = dphysics(z_grid=z_grid, controls=controls, friction=friction)

        # compute the loss as the mean squared error between the predicted and ground truth poses
        loss = physics_loss(states_pred=states_pred, states_gt=states_gt, pred_ts=control_ts, gt_ts=traj_ts,
                            gamma=1., rotation_loss=False)
        # backward pass
        loss.backward()
        optimizer.step()

        print(f'Iteration {i}, Loss: {loss.item()}')
        losses.append(loss.item())

    plt.figure()
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()

    if vis:
        visualize(states=states_pred, x_points=dphysics.x_points, forces=forces_pred,
                  x_grid=dphys_cfg.x_grid[None],
                  y_grid=dphys_cfg.y_grid[None], z_grid=z_grid, states_gt=[X])


def main():
    # optimize_terrain()
    # optimize_terrain_heads()
    optimize_model()


if __name__ == '__main__':
    main()
