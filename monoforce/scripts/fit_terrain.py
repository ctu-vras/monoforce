import sys
sys.path.append('../src')
import numpy as np
import torch
from monoforce.models.traj_predictor.dphysics import DPhysics
from monoforce.models.traj_predictor.dphys_config import DPhysConfig
from monoforce.losses import physics_loss, total_variation
import os
import matplotlib.pyplot as plt


def optimize_terrain():
    from time import time

    dphys_cfg = DPhysConfig(grid_res=0.4)

    # simulation parameters
    T, dt = 6.0, 0.01
    dphys_cfg.dt = dt
    dphys_cfg.traj_sim_time = T
    n_iters = 100
    vis = True

    # heightmap defining the terrain
    x_grid, y_grid = dphys_cfg.x_grid, dphys_cfg.y_grid
    z_grid_gt = torch.exp(-(x_grid - 2.5) ** 2 / 1) * torch.exp(-(y_grid - 0) ** 2 / 4)
    # repeat the heightmap for each rigid body
    z_grid_gt = z_grid_gt.repeat(1, 1, 1)

    # control inputs in m/s
    controls = torch.tensor([[[1.0, 0.0]] * int(dphys_cfg.traj_sim_time / dphys_cfg.dt)])

    # simulate the rigid body dynamics
    dphysics = DPhysics(dphys_cfg)
    states_gt, forces_gt = dphysics(z_grid=z_grid_gt, controls=controls)
    if vis:
        with torch.no_grad():
            dphysics.visualize(states=states_gt, z_grid=z_grid_gt)

    # initial guess for the heightmap
    z_grid = torch.zeros_like(z_grid_gt, requires_grad=True)
    friction = 0.5 * torch.ones_like(z_grid)
    friction.requires_grad = True

    # optimization: height and friction with different learning rates
    optimizer = torch.optim.Adam([{'params': z_grid, 'lr': 0.02},
                                  {'params': friction, 'lr': 0.01}])

    loss_min = np.inf
    z_grid_best = z_grid.clone()
    losses_history = {'traj': [], 'hdiff': [], 'total': []}
    ts = torch.arange(0, T, dt)[None]
    for i in range(n_iters):
        optimizer.zero_grad()

        states, _ = dphysics(z_grid=z_grid, controls=controls, friction=friction)
        loss_traj = physics_loss(states_pred=states, states_gt=states_gt, pred_ts=ts, gt_ts=ts, gamma=0.9)
        loss_terrain = total_variation(z_grid)
        loss = loss_traj #+ loss_terrain

        loss.backward()
        optimizer.step()
        print(f'Iteration {i}, Loss: {loss.item()}')
        # print(f'Heightmap mean: {z_grid.mean().item()}')
        # print(f'Friction mean: {friction.mean().item()}')

        if loss.item() < loss_min:
            loss_min = loss.item()
            z_grid_best = z_grid.clone()

        # visualize the optimized heightmap
        if vis and i % 50 == 0:
            states, forces = dphysics(z_grid=z_grid, controls=controls, friction=friction)
            with torch.no_grad():
                dphysics.visualize(states=states, z_grid=z_grid, states_gt=states_gt, friction=friction)

        losses_history['traj'].append(loss_traj.item())
        losses_history['hdiff'].append(loss_terrain.item())
        losses_history['total'].append(loss.item())

    plt.figure()
    # for key, loss in losses_history.items():
    #     plt.plot(loss, label=key)
    plt.plot(losses_history['traj'], label='traj')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    os.makedirs('./gen/terrain_optimization/', exist_ok=True)
    plt.savefig(f'./gen/terrain_optimization/losses_{time()}.png')
    plt.show()

    # visualize the best heightmap
    states, forces = dphysics(z_grid=z_grid_best, controls=controls, friction=friction)
    with torch.no_grad():
        dphysics.visualize(states=states, z_grid=z_grid, states_gt=states_gt, friction=friction)


def main():
    optimize_terrain()


if __name__ == '__main__':
    main()
