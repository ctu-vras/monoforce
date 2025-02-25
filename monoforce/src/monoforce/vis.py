from matplotlib import pyplot as plt
import numpy as np
from mayavi import mlab


__all__ = [
    'visualize_imgs',
    'set_axes_equal',
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
