import os
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm, pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from PIL import ImageFile
import torch
import open3d as o3d
from mayavi import mlab
from .config import DPhysConfig
ImageFile.LOAD_TRUNCATED_IMAGES = True


__all__ = [
    'set_axes_equal',
    'setup_visualization',
    'animate_trajectory',
    'visualize_imgs',
    'show_cloud',
    'show_cloud_plt',
    'plot_grad_flow',
    'draw_coord_frames',
    'draw_coord_frame',
]

def draw_points_on_image(points, color, image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(points[1][i])), 4, (int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def setup_visualization(system, states=None, states_true=None,
                        cfg=DPhysConfig(), z_margin=0.1, show=False):
    """ Set-up visualization """
    points = system.robot_points.detach().cpu().numpy()
    height = system.height.detach().cpu().numpy()

    h, w = height.shape
    x_grid, y_grid = np.mgrid[-h // 2:h // 2, -w // 2:w // 2] * cfg.grid_res

    mlab.figure(size=(1280, 720))  # bgcolor=(1, 1, 1), fgcolor=(0.2, 0.2, 0.2))
    mlab.clf()
    visu_trajectory = mlab.plot3d
    visu_forces = mlab.quiver3d
    if states is not None:
        pos_x, pos_R, vel_x, vel_omega, forces = [s.detach().cpu().numpy() for s in states]
        visu_trajectory = mlab.plot3d(pos_x[:, 0], pos_x[:, 1], pos_x[:, 2], color=(0, 1, 0), line_width=2.0)
        visu_forces = mlab.quiver3d(points[0, :], points[1, :], points[2, :],
                                    forces[0, 0, :], forces[0, 1, :], forces[0, 2, :],
                                    line_width=4.0, scale_factor=0.005)  # , color=(0.8, 0.8, 0.8))

    if states_true is not None:
        # visualize ground truth trajectory
        pos_x_true, pos_R_true, vel_x_true, vel_omega_true, forces_true = [s.detach().cpu().numpy() for s in states_true]
        mlab.plot3d(pos_x_true[:, 0], pos_x_true[:, 1], pos_x_true[:, 2], color=(0, 0, 1), line_width=2.0)
        poses_gt = np.asarray([np.eye(4) for _ in range(pos_x_true.shape[0])])
        poses_gt[:, :3, 3] = pos_x_true.squeeze()
        poses_gt[:, :3, :3] = pos_R_true
        draw_coord_frames(poses_gt, scale=0.2)
        # draw velocities at ground truth poses locations
        mlab.quiver3d(pos_x_true[:, 0], pos_x_true[:, 1], pos_x_true[:, 2],
                      vel_x_true[:, 0], vel_x_true[:, 1], vel_x_true[:, 2],
                      line_width=2.0, scale_factor=1.)

    visu_rigid_mesh = mlab.mesh(x_grid, y_grid, height,
                                scalars=system.friction.detach().cpu().numpy(), opacity=0.3, vmax=1.0, vmin=0.5,
                                colormap='jet',
                                representation='surface')  # color=(0.15, 0.07, 0.0)
    visu_rigid_wires = mlab.surf(x_grid, y_grid, height, opacity=0.3,
                                 color=(0.6, 0.5, 0.4), representation='wireframe', line_width=5.0)
    visu_robot = mlab.points3d(points[0, :], points[1, :], z_margin + points[2, :], scale_factor=0.25)

    # mlab.colorbar(object=visu_rigid_mesh, title="Terrain friction coefficient")
    mlab.view(azimuth=150, elevation=80, distance=16.0)

    if show:
        mlab.show()
    return visu_robot, visu_forces, visu_trajectory, visu_rigid_mesh, visu_rigid_wires


def animate_trajectory(system, vis_cfg, z_margin=0., frame_n=0, log_path='./gen', save_figs_step=5):
    visu_robot, visu_forces, visu_trajectory, visu_rigid_mesh, visu_rigid_wires = vis_cfg

    points = system.robot_points.detach().cpu().numpy()
    h_r = system.height.detach().cpu().numpy()
    h = h_r + system.height_soft.detach().cpu().numpy()
    visu_rigid_mesh.mlab_source.z = np.asarray(h_r, 'd')
    visu_rigid_wires.mlab_source.scalars = np.asarray(h_r, 'd')
    # visu_soft_mesh.mlab_source.z = np.asarray(h, 'd')
    # visu_soft_wires.mlab_source.scalars = np.asarray(h, 'd')
    # visu_soft_mesh.mlab_source.scalars = np.asarray(system.friction.detach().cpu().numpy(), 'd')
    visu_rigid_mesh.mlab_source.scalars = np.asarray(system.friction.detach().cpu().numpy(), 'd')
    # visu_rigid_mesh.mlab_source.scalars = np.asarray(system.elasticity.detach().cpu().numpy(), 'd')

    visu_rigid_wires.mlab_source.set(scalars=np.asarray(system.height.detach().cpu().numpy(), 'd'),
                                     z=np.asarray(system.friction.detach().cpu().numpy(), 'd'))
    for t in range(system.pos_x.shape[0]):
        system.cog = system.pos_x[t]
        system.rot_p = system.pos_R[t] @ points
        system.forces_p = system.forces[t]
        visu_robot.mlab_source.set(x=system.cog[0] + system.rot_p[0, :],
                                   y=system.cog[1] + system.rot_p[1, :],
                                   z=z_margin + system.cog[2] + system.rot_p[2, :])
        # visu_robot.mlab_source.scalars = ((system.forces_p**2).sum(axis=0) > 0).astype(int) + 1
        visu_forces.mlab_source.set(x=system.cog[0] + system.rot_p[0, :],
                                    y=system.cog[1] + system.rot_p[1, :],
                                    z=z_margin + system.cog[2] + system.rot_p[2, :],
                                    u=system.forces_p[0, :],
                                    v=system.forces_p[1, :], w=system.forces_p[2, :])
        visu_trajectory.mlab_source.set(x=system.pos_x[:, 0].squeeze(),
                                        y=system.pos_x[:, 1].squeeze(),
                                        z=system.pos_x[:, 2].squeeze())
        # mlab.view(azimuth=150 - frame_n, elevation=60, distance=16.0)

        if t % save_figs_step == 0:
            os.makedirs(log_path, exist_ok=True)
            mlab.savefig(filename=os.path.join(log_path, '{:04d}_frame.png'.format(frame_n)), magnification=1.0)
            frame_n += 1

    return frame_n

# helper function for data visualization
def visualize_imgs(layout='rows', figsize=(20, 10), **images):
    assert layout in ['columns', 'rows']
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=figsize)
    for i, (name, image) in enumerate(images.items()):
        if layout == 'rows':
            plt.subplot(1, n, i + 1)
        elif layout == 'columns':
            plt.subplot(n, 1, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.tight_layout()
    plt.show()


def map_colors(values, colormap=cm.gist_rainbow, min_value=None, max_value=None):
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
    assert callable(colormap) or isinstance(colormap, torch.Tensor)
    if min_value is None:
        min_value = values[torch.isfinite(values)].min()
    if max_value is None:
        max_value = values[torch.isfinite(values)].max()
    scale = max_value - min_value
    a = (values - min_value) / scale if scale > 0.0 else values - min_value
    if callable(colormap):
        colors = colormap(a.squeeze())[:, :3]
        return colors
    # TODO: Allow full colormap with multiple colors.
    assert isinstance(colormap, torch.Tensor)
    num_colors = colormap.shape[0]
    a = a.reshape([-1, 1])
    if num_colors == 2:
        # Interpolate the two colors.
        colors = (1 - a) * colormap[0:1] + a * colormap[1:]
    else:
        # Select closest based on scaled value.
        i = torch.round(a * (num_colors - 1))
        colors = colormap[i]
    return colors


def show_cloud(x, value=None, min=None, max=None, colormap=cm.jet):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    if value is not None:
        assert isinstance(value, np.ndarray)
        if value.ndim == 2:
            assert value.shape[1] == 3
            colors = value
        elif value.ndim == 1:
            colors = map_colors(value, colormap=colormap, min_value=min, max_value=max)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def show_cloud_plt(P, **kwargs):
    if P.dtype.names:
        P = structured_to_unstructured(P[['x', 'y', 'z']])

    ax = plt.axes(projection='3d')

    ax.plot(P[:, 0], P[:, 1], P[:, 2], 'o', **kwargs)

    set_axes_equal(ax)
    ax.grid()


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


def plot_grad_flow(named_parameters, max_grad_vis=None):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    if max_grad_vis:
        plt.ylim(bottom=-0.001, top=max_grad_vis)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


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


def vis_dem_data(height, traj, img=None, cfg=DPhysConfig()):
    assert len(height.shape) == 2, 'Height map should be 2D'
    assert len(traj['poses'].shape) == 3, 'Trajectory should be 3D'
    plt.figure(figsize=(20, 10))
    # add subplot
    ax = plt.subplot(131)
    ax.set_title('Height map')
    ax.imshow(height.T, cmap='jet', origin='lower', vmin=-1, vmax=1)
    x, y = traj['poses'][:, 0, 3], traj['poses'][:, 1, 3]
    h, w = height.shape
    x_grid, y_grid = x / cfg.grid_res + w / 2, y / cfg.grid_res + h / 2
    plt.plot(x_grid, y_grid, 'rx-', label='Robot trajectory')
    time_ids = np.linspace(1, len(x_grid), len(x_grid), dtype=int)
    # plot time indices for each waypoint in a trajectory
    for i, txt in enumerate(time_ids):
        ax.annotate(txt, (x_grid[i], y_grid[i]))
    plt.legend()

    # visualize heightmap as a surface in 3D
    X = np.arange(-cfg.d_max, cfg.d_max, cfg.grid_res)
    Y = np.arange(-cfg.d_max, cfg.d_max, cfg.grid_res)
    X, Y = np.meshgrid(X, Y)
    Z = height
    ax = plt.subplot(132, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Height map')
    set_axes_equal(ax)

    if img is not None:
        # visualize image
        img = np.asarray(img)
        ax = plt.subplot(133)
        ax.imshow(img)
        ax.set_title('Camera view')
        ax.axis('off')

    plt.show()
