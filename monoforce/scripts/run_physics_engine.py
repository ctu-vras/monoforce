#!/usr/bin/env python

import sys
sys.path.append('../src')
import numpy as np
import torch
from monoforce.models.physics_engine.configs import (
    WorldConfig,
    RobotModelConfig,
    PhysicsEngineConfig,
)
from monoforce.models.physics_engine.engine.engine import DPhysicsEngine, PhysicsState
from monoforce.models.physics_engine.utils.geometry import euler_to_quaternion
from monoforce.models.physics_engine.vis.animator import animate_trajectory
from collections import deque


def motion():
    num_robots = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Heightmap setup - use torch's XY indexing !!!!!
    grid_res = 0.1  # 5cm per grid cell
    max_coord = 6.4  # meters
    DIM = int(2 * max_coord / grid_res)
    xint = torch.linspace(-max_coord, max_coord, DIM)
    yint = torch.linspace(-max_coord, max_coord, DIM)
    x_grid, y_grid = torch.meshgrid(xint, yint, indexing="xy")
    z_grid = torch.exp(-(x_grid - 2) ** 2 / 2) * torch.exp(-(y_grid - 0) ** 2 / 4)
    x_grid = x_grid.repeat(num_robots, 1, 1)
    y_grid = y_grid.repeat(num_robots, 1, 1)
    z_grid = z_grid.repeat(num_robots, 1, 1)

    # Instantiate the configs
    robot_model = RobotModelConfig(kind="marv")
    world_config = WorldConfig(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        grid_res=grid_res,
        max_coord=max_coord,
        k_stiffness=40000,
    )
    physics_config = PhysicsEngineConfig(num_robots=num_robots)

    # Controls
    traj_length = 10.0  # seconds
    n_iters = int(traj_length / physics_config.dt)
    speed = 1.0  # m/s forward
    omega = 0.0  # rad/s yaw
    controls = robot_model.vw_to_vels(speed, omega)
    flipper_controls = torch.zeros_like(controls)
    controls_all = torch.cat((controls, flipper_controls), dim=-1).repeat(n_iters, num_robots, 1).to(device)
    # Set joint rotational velocities
    amplitude = 1. * torch.pi / 4
    periods = traj_length / 10.0
    rot_vels = torch.cos(torch.linspace(0, periods * 2 * np.pi, n_iters)) * amplitude
    rot_vels = rot_vels.unsqueeze(-1).repeat(1, num_robots)
    controls_all[:, :, robot_model.num_driving_parts] = rot_vels
    controls_all[:, :, robot_model.num_driving_parts + 1] = rot_vels
    controls_all[:, :, robot_model.num_driving_parts + 2] = -rot_vels
    controls_all[:, :, robot_model.num_driving_parts + 3] = -rot_vels

    for cfg in [robot_model, world_config, physics_config]:
        cfg.to(device)

    # Instantiate the physics engine
    engine = DPhysicsEngine(physics_config, robot_model, device)

    # Initial state
    x0 = torch.tensor([-1.0, 0.0, 0.1]).to(device).repeat(num_robots, 1)
    xd0 = torch.zeros_like(x0)
    q0 = euler_to_quaternion(*torch.tensor([0, 0, 0.0 * torch.pi])).to(device).repeat(num_robots, 1)
    omega0 = torch.zeros_like(x0)
    thetas0 = torch.zeros(num_robots, robot_model.num_driving_parts).to(device)
    init_state = PhysicsState(x0, xd0, q0, omega0, thetas0)

    states = deque(maxlen=n_iters)
    auxs = deque(maxlen=n_iters)

    state = init_state
    for i in range(n_iters):
        state, der, aux = engine(state, controls_all[i], world_config)
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
    motion()


if __name__ == '__main__':
    main()
