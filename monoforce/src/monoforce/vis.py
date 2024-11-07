import os
from matplotlib import cm, pyplot as plt
import numpy as np
from mayavi import mlab


__all__ = [
    'visualize_imgs',
    'set_axes_equal',
    'setup_visualization',
    'animate_trajectory',
    'draw_coord_frames',
    'draw_coord_frame',
]

def visualize_imgs(images, names=None):
    n = len(images)
    figsize = (n * 5, 5)
    plt.figure(figsize=figsize)
    for i, image in enumerate(images):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
        if names is not None:
            assert len(names) >= n, f'Number of names {len(names)} must not be smaller than the number of images {n}'
            plt.title(names[i])
    plt.tight_layout()
    plt.show()


def setup_visualization(states, forces, x_grid, y_grid, z_grid, states_gt=None):
    # unpack the states and forces
    xs, x_points = states[0], states[4]
    F_spring, F_friction = forces
    assert xs.shape[1] == 3, 'States should be 3D'
    assert x_points.shape[2] == 3, 'Points should be 3D'
    assert F_spring.shape == F_friction.shape == x_points.shape, 'Forces should have the same shape as points'

    # set up the visualization
    mlab.figure(size=(1280, 720))
    mlab.clf()
    visu_traj = mlab.plot3d(xs[:, 0], xs[:, 1], xs[:, 2], color=(0, 1, 0), line_width=2.0)
    # visu_Ns = mlab.quiver3d(x_points[0, :, 0], x_points[0, :, 1], x_points[0, :, 2],
    #                         F_spring[0, :, 0], F_spring[0, :, 1], F_spring[0, :, 2],
    #                         line_width=1.0, scale_factor=0.1, color=(0, 0, 1))
    # visu_Frs = mlab.quiver3d(x_points[0, :, 0], x_points[0, :, 1], x_points[0, :, 2],
    #                          F_friction[0, :, 0], F_friction[0, :, 1], F_friction[0, :, 2],
    #                          line_width=1.0, scale_factor=1.0, color=(0, 1, 0))
    visu_Ns, visu_Frs = None, None
    visu_terrain = mlab.mesh(x_grid, y_grid, z_grid, colormap='terrain', opacity=0.6)
    visu_robot = mlab.points3d(x_points[0, :, 0], x_points[0, :, 1], x_points[0, :, 2],
                               scale_factor=0.03, color=(0, 0, 0))

    visu_cfg = [visu_traj, visu_Ns, visu_Frs, visu_terrain, visu_robot]

    if states_gt:
        xs_gt = states_gt[0]
        assert xs_gt.shape[1] == 3, 'States should be 3D'
        visu_traj_gt = mlab.plot3d(xs_gt[:, 0], xs_gt[:, 1], xs_gt[:, 2], color=(0, 0, 1), line_width=2.0)
        visu_cfg.append(visu_traj_gt)

    # set view angle: top down from 10 units above
    mlab.view(azimuth=0, elevation=0, distance=15)

    return visu_cfg


def animate_trajectory(states, forces, z_grid, vis_cfg, step=1, friction=None):
    # unpack the states and forces
    xs, xds, rs, omegas, x_points = states
    F_spring, F_friction = forces
    assert xs.shape[1] == 3, 'States should be 3D'
    assert xds.shape[1] == 3, 'Velocities should be 3D'
    assert rs.shape[-2:] == (3, 3), 'Rotations should be 3x3'
    assert omegas.shape[1] == 3, 'Angular velocities should be 3D'
    assert x_points.shape[2] == 3, 'Points should be 3D'
    assert F_spring.shape == F_friction.shape == x_points.shape, 'Forces should have the same shape as points'

    # unpack the visualization configuration
    visu_traj, visu_Ns, visu_Frs, visu_terrain, visu_robot = vis_cfg[:5]

    # plot the terrain
    visu_terrain.mlab_source.z = z_grid
    if friction is not None:
        visu_terrain.mlab_source.scalars = friction

    # plot the trajectory
    visu_traj.mlab_source.set(x=xs[:, 0], y=xs[:, 1], z=xs[:, 2])

    # animate robot's motion and forces
    for t in range(len(xs)):
        visu_robot.mlab_source.set(x=x_points[t, :, 0], y=x_points[t, :, 1], z=x_points[t, :, 2])
        # visu_Ns.mlab_source.set(x=x_points[t, :, 0], y=x_points[t, :, 1], z=x_points[t, :, 2],
        #                         u=F_spring[t, :, 0], v=F_spring[t, :, 1], w=F_spring[t, :, 2])
        # visu_Frs.mlab_source.set(x=x_points[t, :, 0], y=x_points[t, :, 1], z=x_points[t, :, 2],
        #                          u=F_friction[t, :, 0], v=F_friction[t, :, 1],
        #                          w=F_friction[t, :, 2])
        if t % step == 0:
            path = os.path.join(os.path.dirname(__file__), '../gen/robot_control')
            os.makedirs(path, exist_ok=True)
            mlab.savefig(f'{path}/frame_{t}.png')
    mlab.show()


# https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def draw_coord_frame(pose, scale=0.5):
    t, R = pose[:3, 3], pose[:3, :3]
    # draw coordinate frame
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    x_axis = R.dot(x_axis)
    y_axis = R.dot(y_axis)
    z_axis = R.dot(z_axis)
    mlab.quiver3d(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2], color=(1, 0, 0), scale_factor=scale)
    mlab.quiver3d(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2], color=(0, 1, 0), scale_factor=scale)
    mlab.quiver3d(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2], color=(0, 0, 1), scale_factor=scale)

def draw_coord_frames(poses, scale=0.1):
    assert poses.ndim == 3
    assert poses.shape[1:] == (4, 4)

    for pose in poses:
        draw_coord_frame(pose, scale=scale)


if __name__ == '__main__':
    from scipy.spatial.transform import Rotation

    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[:3, 3] = np.array([0, 2, 0])
    T1[:3, :3] = Rotation.from_euler('xyz', [45, 0, 0], degrees=True).as_matrix()

    # Create a figure
    fig = mlab.figure()
    draw_coord_frame(T0)
    draw_coord_frame(T1)
    # Show the figure
    mlab.show()

