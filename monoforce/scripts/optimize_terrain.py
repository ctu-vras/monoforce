import sys
sys.path.append('../')
import torch
from monoforce.models.physics_engine.engine.engine import DPhysicsEngine, PhysicsState
from monoforce.configs import WorldConfig, RobotModelConfig, PhysicsEngineConfig
from monoforce.losses import trajectory_loss
from monoforce.models.physics_engine.engine.engine_state import vectorize_iter_of_states as vectorize_states
from monoforce.models.physics_engine.vis.animator import animate_trajectory
from monoforce.models.physics_engine.utils.environment import make_x_y_grids
from monoforce.utils import set_device
from collections import deque
import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.use('Qt5Agg')


def optimize_terrain():
    # simulation parameters
    n_opt_iters = 100
    device = set_device('cuda')
    vis = False
    n_robots = 1

    # Heightmap setup
    grid_res = 0.1  # meters per grid cell
    max_coord = 6.4  # meters
    x_grid, y_grid = make_x_y_grids(max_coord, grid_res, n_robots)
    z_grid = torch.zeros_like(x_grid, requires_grad=True, device=device)

    # Instantiate the configs
    robot_cfg = RobotModelConfig()
    world_config = WorldConfig(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        grid_res=grid_res,
        max_coord=max_coord,
    )
    physics_config = PhysicsEngineConfig(num_robots=n_robots)
    for cfg in [robot_cfg, world_config, physics_config]:
        cfg.to(device)

    # Instantiate the physics engine
    engine = DPhysicsEngine(physics_config, robot_cfg, device)

    # Controls
    T = 4.0  # seconds
    dt = physics_config.dt
    speed = 1. * torch.ones(n_robots, device=device)  # m/s forward
    omega = torch.zeros(n_robots, device=device)
    flipper_vels = robot_cfg.vw_to_vels(speed, omega)
    flipper_omegas = torch.zeros_like(flipper_vels)
    controls = torch.cat((flipper_vels, flipper_omegas), dim=-1).repeat(int(T / dt), 1, 1)

    # Initial state
    x0 = torch.tensor([0., 0., 0.]).repeat(n_robots, 1)
    xd0 = torch.zeros_like(x0)
    q0 = torch.tensor([1., 0., 0., 0.]).repeat(n_robots, 1)
    omega0 = torch.zeros_like(x0)
    thetas0 = torch.zeros(robot_cfg.num_driving_parts).repeat(n_robots, 1)
    state0 = PhysicsState(x0, xd0, q0, omega0, thetas0, batch_size=n_robots).to(device)

    # optimization: height and friction with different learning rates
    optimizer = torch.optim.Adam([{'params': z_grid, 'lr': 0.01}])

    losses_history = []
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ts = torch.arange(0, T, dt).to(device)
    x_gt = torch.cat([
        torch.linspace(0, T, int(T / dt)).unsqueeze(-1),
        torch.zeros(int(T / dt)).unsqueeze(-1),
        torch.linspace(0, 0.5, int(T / dt)).unsqueeze(-1),
    ], dim=1).repeat(n_robots, 1, 1).to(device)
    ts = ts.repeat(n_robots, 1)
    for i in range(n_opt_iters):
        optimizer.zero_grad()

        state = state0.copy()
        states = deque(maxlen=int(T / dt))
        auxs = deque(maxlen=int(T / dt))
        world_config.z_grid = z_grid
        for t in range(int(T / dt)):
            state, der, aux = engine(state, controls[t], world_config)
            states.append(state)
            auxs.append(aux)
        states_vec = vectorize_states(states)
        loss = trajectory_loss(x_pred=states_vec.x.permute(1, 0, 2), x_gt=x_gt, pred_ts=ts, gt_ts=ts, gamma=0.0)

        loss.backward()
        optimizer.step()
        print(f'Iteration {i}, Loss: {loss.item()}')
        losses_history.append(loss.item())

        with torch.no_grad():
            if vis and i % 10 == 0:
                # visualize the trajectory
                animate_trajectory(
                    world_config,
                    physics_config,
                    states,
                    auxs,
                )

            for ax in axes:
                ax.cla()
            axes[0].plot(losses_history, 'k')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Loss')
            axes[0].grid()

            # plot trajectories
            x_pred_vis = states_vec.x.permute(1, 0, 2).cpu().numpy()[0]
            x_gt_vis = x_gt.cpu().numpy()[0]
            ts_vis = torch.arange(0, T, dt).cpu().numpy()
            axes[1].plot(ts_vis, x_pred_vis[:, 0], 'r', label='X(t)')
            axes[1].plot(ts_vis, x_pred_vis[:, 1], 'g', label='Y(t)')
            axes[1].plot(ts_vis, x_pred_vis[:, 2], 'b', label='Z(t)')
            axes[1].plot(ts_vis, x_gt_vis[:, 0], 'r--', label='X_gt(t)')
            axes[1].plot(ts_vis, x_gt_vis[:, 1], 'g--', label='Y_gt(t)')
            axes[1].plot(ts_vis, x_gt_vis[:, 2], 'b--', label='Z_gt(t)')
            axes[1].set_ylabel('position, [m]')
            axes[1].set_xlabel('t, [s]')
            axes[1].set_title('Trajectories')
            axes[1].grid()
            axes[1].legend()

            plt.pause(0.01)
            plt.draw()

    plt.show()


def main():
    optimize_terrain()


if __name__ == '__main__':
    main()