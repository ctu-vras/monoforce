#!/usr/bin/env python

import sys
sys.path.append('../src')
import torch
from monoforce.configs import (
    WorldConfig,
    RobotModelConfig,
    PhysicsEngineConfig,
)
from monoforce.models.physics_engine.engine.engine import DPhysicsEngine, PhysicsState
from monoforce.models.physics_engine.utils.geometry import euler_to_quaternion
from monoforce.models.physics_engine.engine.engine_state import vectorize_iter_of_states as vectorize_states
from monoforce.models.physics_engine.vis.animator import animate_trajectory
from collections import deque
import pyvista as pv
import numpy as np


def motion():
    n_robots = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Heightmap setup
    grid_res = 0.1  # 10cm per grid cell
    max_coord = 6.4  # meters
    DIM = int(2 * max_coord / grid_res)
    xint = torch.linspace(-max_coord, max_coord, DIM)
    yint = torch.linspace(-max_coord, max_coord, DIM)
    x_grid, y_grid = torch.meshgrid(xint, yint, indexing="xy")  # use torch's XY indexing !!!!!
    z_grid = 0.5 * torch.exp(-(x_grid - 2) ** 2 / 1) * torch.exp(-(y_grid - 0) ** 2 / 2)
    x_grid = x_grid.repeat(n_robots, 1, 1)
    y_grid = y_grid.repeat(n_robots, 1, 1)
    z_grid = z_grid.repeat(n_robots, 1, 1)

    # Instantiate the configs
    robot_model = RobotModelConfig(kind="marv")
    world_config = WorldConfig(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        grid_res=grid_res,
        max_coord=max_coord,
    )
    physics_config = PhysicsEngineConfig(num_robots=n_robots)

    # Controls
    traj_length = 5.0  # seconds
    n_iters = int(traj_length / physics_config.dt)
    speed = 1. * torch.ones(n_robots)  # m/s forward
    speed[::2] = -speed[::2]
    omega = torch.linspace(-1., 1., n_robots)  # rad/s yaw
    flipper_vels = robot_model.vw_to_vels(speed, omega)
    flipper_omegas = torch.zeros_like(flipper_vels)
    controls = torch.cat((flipper_vels, flipper_omegas), dim=-1).repeat(n_iters, 1, 1).to(device)

    for cfg in [robot_model, world_config, physics_config]:
        cfg.to(device)

    # Instantiate the physics engine
    engine = DPhysicsEngine(physics_config, robot_model, device)

    # Initial state
    x0 = torch.tensor([-1.0, 0.0, 0.1]).to(device).repeat(n_robots, 1)
    xd0 = torch.zeros_like(x0)
    q0 = euler_to_quaternion(*torch.tensor([0, 0, 0.0 * torch.pi])).to(device).repeat(n_robots, 1)
    omega0 = torch.zeros_like(x0)
    thetas0 = torch.zeros(n_robots, robot_model.num_driving_parts).to(device)
    init_state = PhysicsState(x0, xd0, q0, omega0, thetas0, batch_size=n_robots)

    states = deque(maxlen=n_iters)
    auxs = deque(maxlen=n_iters)

    state = init_state
    for i in range(n_iters):
        state, der, aux = engine(state, controls[i], world_config)
        states.append(state)
        auxs.append(aux)

    states_vec = vectorize_states(states)
    print(states_vec.x.shape)

    # visualization
    animate_trajectory(
        world_config,
        physics_config,
        states,
        auxs,
        robot_index=np.random.randint(0, n_robots-1),
    )

    # plot terrain surface with pyvista
    plotter = pv.Plotter()
    X, Y, Z = world_config.x_grid[0].cpu().numpy(), world_config.y_grid[0].cpu().numpy(), world_config.z_grid[0].cpu().numpy()
    surface = pv.StructuredGrid(X, Y, Z)
    plotter.add_mesh(surface, cmap="terrain")
    # plot trajectory
    for i in range(n_robots):
        xs = states_vec.x[:, i].cpu().numpy()
        plotter.add_lines(xs, width=5, color='blue')
    plotter.show()


def motion_rough():
    from monoforce.datasets.rough import ROUGH, rough_seq_paths
    from monoforce.utils import explore_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq = rough_seq_paths[0]
    ds = ROUGH(seq)
    sample_i = np.random.randint(0, len(ds))
    sample = ds[sample_i]
    explore_data(ds, sample_range=[sample_i])
    (imgs, rots, trans, intrins, post_rots, post_trans,
     hm_geom, hm_terrain,
     control_ts, controls,
     traj_ts, xs, xds, qs, omegas, thetas) = sample

    n_iters = len(control_ts)
    d_max, grid_res = 6.4, 0.1
    xint = torch.linspace(-d_max, d_max, int(2 * d_max / grid_res))
    yint = torch.linspace(-d_max, d_max, int(2 * d_max / grid_res))
    x_grid, y_grid = torch.meshgrid(xint, yint, indexing="xy")
    z_grid = hm_geom[0]

    # Instantiate the configs
    n_robots = 1
    robot_model = RobotModelConfig(kind="marv")
    world_config = WorldConfig(
        x_grid=x_grid.repeat(n_robots, 1, 1),
        y_grid=y_grid.repeat(n_robots, 1, 1),
        z_grid=z_grid.repeat(n_robots, 1, 1),
        grid_res=grid_res,
        max_coord=d_max,
    )
    physics_config = PhysicsEngineConfig(num_robots=n_robots)
    for cfg in [robot_model, world_config, physics_config]:
        cfg.to(device)

    # Instantiate the physics engine
    engine = DPhysicsEngine(physics_config, robot_model, device)

    # Initial state
    x0 = xs[0].repeat(n_robots, 1)
    xd0 = torch.zeros_like(x0)
    q0 = qs[0].repeat(n_robots, 1)
    omega0 = torch.zeros_like(x0)
    thetas0 = thetas[0].repeat(n_robots, 1)
    state0 = PhysicsState(x0, xd0, q0, omega0, thetas0, batch_size=n_robots).to(device)

    states = deque(maxlen=n_iters)
    auxs = deque(maxlen=n_iters)
    state = state0
    controls = controls.repeat(n_robots, 1, 1).to(device)
    for i in range(n_iters):
        u = controls[:, i]
        state, der, aux = engine(state, u, world_config)
        states.append(state)
        auxs.append(aux)

    # visualization
    animate_trajectory(
        world_config,
        physics_config,
        states,
        auxs,
    )


def main():
    # motion()
    motion_rough()


if __name__ == '__main__':
    main()