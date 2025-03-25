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
from monoforce.models.physics_engine.utils.geometry import unit_quaternion, euler_to_quaternion
from monoforce.models.physics_engine.engine.engine_state import (
    vectorize_iter_of_states as vectorize_states,
)
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
    x, y = torch.meshgrid(xint, yint, indexing="xy")

    # block hm
    z = torch.zeros_like(x)
    # for thresh in [1.0, 0, -1.0, -2]:
    #     z[torch.logical_and(x > -thresh, y < thresh)] += 0.2
    x_grid = x.repeat(num_robots, 1, 1)
    y_grid = y.repeat(num_robots, 1, 1)
    z_grid = z.repeat(num_robots, 1, 1)

    # Instatiate the physics config
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
    traj_length = 4.0  # seconds
    n_iters = int(traj_length / physics_config.dt)
    speed = 1.0  # m/s forward
    omega = 0.0  # rad/s yaw
    controls = robot_model.vw_to_vels(speed, omega)
    flipper_controls = torch.zeros_like(controls)

    for cfg in [robot_model, world_config, physics_config]:
        cfg.to(device)

    engine = DPhysicsEngine(physics_config, robot_model, device)

    x0 = torch.tensor([0.0, 0.0, 0.1]).to(device).repeat(num_robots, 1)
    xd0 = torch.zeros_like(x0)
    q0 = euler_to_quaternion(*torch.tensor([0, 0, 0.0 * torch.pi])).to(device).repeat(num_robots, 1)
    omega0 = torch.zeros_like(x0)
    thetas0 = torch.zeros(num_robots, robot_model.num_driving_parts).to(device)
    controls_all = torch.cat((controls, flipper_controls), dim=-1).repeat(n_iters, num_robots, 1).to(device)

    # Set joint rotational velocities, we want to follow a sine wave, so we set the joint velocities to the derivative of the sine wave
    # We want to go +- pi/6 5 times in 10 seconds
    amplitude = 0 * torch.pi / 4
    periods = traj_length / 10.0
    rot_vels = torch.cos(torch.linspace(0, periods * 2 * np.pi, n_iters)) * amplitude
    rot_vels = rot_vels.unsqueeze(-1).repeat(1, num_robots)
    controls_all[:, :, robot_model.num_driving_parts] = rot_vels
    controls_all[:, :, robot_model.num_driving_parts + 1] = rot_vels
    controls_all[:, :, robot_model.num_driving_parts + 2] = -rot_vels
    controls_all[:, :, robot_model.num_driving_parts + 3] = -rot_vels

    init_state = PhysicsState(x0, xd0, q0, omega0, thetas0)

    states = deque(maxlen=n_iters)
    dstates = deque(maxlen=n_iters)
    auxs = deque(maxlen=n_iters)

    state = init_state
    for i in range(n_iters):
        state, der, aux = engine(state, controls_all[i], world_config)
        states.append(state)
        dstates.append(der)
        auxs.append(aux)

    states_vec = vectorize_states(states)
    dstates_vec = vectorize_states(dstates)
    aux_vec = vectorize_states(auxs)


def main():
    motion()


if __name__ == '__main__':
    main()
