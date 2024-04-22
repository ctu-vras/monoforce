import torch
import numpy as np
from .transformations import rot2rpy
import time


__all__ = [
    'keyboard_controller',
    'velocity_controller',
    'pose_controller',
    'pose_control',
    'cmd_vel_from_goal'
]


def keyboard_controller(system, key):
    # assert isinstance(system, RigidBodySoftTerrain)

    # print('\nYou Entered {0}'.format(key))
    vel_step = 0.1
    try:
        if key.char == 'w':
            system.vel_tracks += vel_step
        if key.char == 'x':
            system.vel_tracks -= vel_step
        if key.char == 'a':
            system.vel_tracks[0] -= vel_step
            system.vel_tracks[1] += vel_step
        if key.char == 'd':
            system.vel_tracks[0] += vel_step
            system.vel_tracks[1] -= vel_step
    except AttributeError:
        system.vel_tracks[0] = 0
        system.vel_tracks[1] = 0
    system.vel_tracks = torch.clip(system.vel_tracks, min=-2, max=2)
    print('current track velocities:', system.vel_tracks.detach().cpu().numpy())

    if key == keyboard.Key.delete:
        # Stop listener
        return False


def velocity_controller(system, cmd_vels, rate=10., vel_max=2.):
    """
    Open-loop velocity controller for a two-tracks robot model.
    Function updates tracks velocities (system.vel_tracks) with velocities from cmd_vels.

    :param system: RigidBodySoftTerrain
    :param cmd_vels: dict with keys 'linear', 'angular', 'stamps'
    :param rate: float
    :param vel_max: float
    """
    # assert isinstance(system, RigidBodySoftTerrain)
    assert isinstance(cmd_vels, dict)
    assert 'linear' in cmd_vels.keys()
    assert 'angular' in cmd_vels.keys()
    assert 'stamps' in cmd_vels.keys()
    N = cmd_vels['linear'].shape[0]
    assert cmd_vels['linear'].shape == (N, 3)
    assert cmd_vels['angular'].shape == (N, 1)
    assert cmd_vels['stamps'].shape == (N, 1)

    goal_i = 0
    tracks_distance = system.robot_points[1].max() - system.robot_points[1].min()
    t = 0.
    dt = 1. / rate
    while t < cmd_vels['stamps'][-1]:
        v = cmd_vels['linear'][goal_i].squeeze().norm()
        w = cmd_vels['angular'][goal_i].squeeze()

        # two tracks robot model
        u1 = v + w * tracks_distance / 4.
        u2 = v - w * tracks_distance / 4.

        system.vel_tracks = torch.tensor([u1, u2], device=system.device)
        system.vel_tracks = torch.clip(system.vel_tracks, min=-vel_max, max=vel_max)

        t += dt
        if t > cmd_vels['stamps'][goal_i+1]:
            goal_i += 1

        time.sleep(dt)
    else:
        print('Done!')
        system.vel_tracks = torch.tensor([0., 0.], device=system.device)


def pose_control(state, goal_pose, Kp_rho=1., Kp_theta=1., Kp_yaw=1.,
                 return_dist=False, allow_backwards=True, dist_reached=0.01):
    # assert isinstance(state, State)
    assert isinstance(goal_pose, torch.Tensor)
    assert goal_pose.shape == (4, 4)

    goal_xyz = goal_pose[:3, 3]
    # distances to goal in robot frame
    d_xyz = goal_xyz.squeeze() - state[0].squeeze()
    dist_xy = torch.linalg.norm(d_xyz[:2])

    if dist_xy < dist_reached:
        # print('Reached goal')
        if return_dist:
            return 0., 0., dist_xy
        return 0., 0.

    # robot heading to goal error
    yaw = rot2rpy(state[1])[2].item()
    d_theta = torch.atan2(d_xyz[1], d_xyz[0]) - yaw
    # d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
    d_theta = torch.as_tensor(d_theta)
    d_theta = torch.atan2(torch.sin(d_theta), torch.cos(d_theta))

    # robot yaw (orientation at goal point) error
    yaw_goal = rot2rpy(goal_pose[:3, :3])[2].item()
    d_yaw = yaw_goal - yaw
    # d_yaw = (d_yaw + np.pi) % (2 * np.pi) - np.pi
    d_yaw = torch.as_tensor(d_yaw)
    d_yaw = torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw))

    if allow_backwards and torch.abs(d_theta) > np.pi / 2.:
        # print('Going backwards')
        d_theta = (d_theta + np.pi / 2.) % np.pi - np.pi / 2.
        vel_sign = -1.
    else:
        vel_sign = 1.

    v = vel_sign * Kp_rho * dist_xy
    w = Kp_theta * d_theta + Kp_yaw * d_yaw

    if return_dist:
        return v, w, dist_xy

    return v, w


def pose_controller(system, goal_poses, rate=10., vel_max=2.5,
                    Kp_rho=1., Kp_theta=1., Kp_yaw=1.,
                    stationary_std=0.01, stationary_time=1, reached_dist=0.2):
    N = goal_poses.shape[0]
    assert goal_poses.shape == (N, 4, 4)

    goal_i = 1
    dists_history = []
    max_iters = int(10 * rate)
    k = 0
    tracks_distance = system.robot_points[1].max() - system.robot_points[1].min()
    while goal_i < len(goal_poses):
        goal_pose = goal_poses[goal_i]

        v, w, rho = pose_control(system.state, goal_pose, Kp_rho, Kp_theta, Kp_yaw, return_dist=True)

        # two tracks robot model
        u1 = v - w * tracks_distance / 4.
        u2 = v + w * tracks_distance / 4.

        system.vel_tracks = torch.tensor([u1, u2], device=system.device)
        system.vel_tracks = torch.clip(system.vel_tracks, min=-vel_max, max=vel_max)

        # if not moving much switch to next goal
        dists_history.append(rho.item())
        k += 1
        if len(dists_history) > stationary_time * int(rate):
            if np.std(dists_history[-stationary_time * int(rate):]) < stationary_std and \
                    rho < reached_dist or k > max_iters:
                # print('Switching to next goal number %i. Reached in %i iters.' % (goal_i + 1, k))
                goal_i += 1
                k = 0
                dists_history = []

        time.sleep(1 / rate)
    else:
        print('Done!')
        system.vel_tracks = torch.tensor([0., 0.], device=system.device)


def cmd_vel_from_goal(x, y, yaw, x_g, y_g, T):
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    heading = torch.atan2(y_g - y, x_g - x)
    omega = 2 * (heading - yaw) / T

    if torch.isclose(omega, torch.tensor(0.)):
        # print('omega is close to zero')
        vel = (x_g - x) / T
    else:
        vel = omega * (x_g - x) /\
              (2 * torch.sin(omega * T / 2) * torch.cos(yaw + omega * T / 2))

    return vel, omega
