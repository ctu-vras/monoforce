import os.path

import torch
from .segmentation import affine
from .transformations import rot2rpy, rpy2rot, transform_cloud
from .utils import timing, position
from .vis import show_cloud
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

default_rng = np.random.default_rng(135)


def filter_range(cloud, min, max, log=False, only_mask=False):
    """Keep points within range interval."""
    assert isinstance(cloud, np.ndarray), type(cloud)
    assert isinstance(min, (float, int)), min
    assert isinstance(max, (float, int)), max
    assert min <= max, (min, max)
    min = float(min)
    max = float(max)
    if min <= 0.0 and max == np.inf:
        return cloud
    if cloud.dtype.names:
        cloud = cloud.ravel()
    x = position(cloud)
    r = np.linalg.norm(x, axis=1)
    mask = (min <= r) & (r <= max)

    if log:
        print('%.3f = %i / %i points kept (range min %s, max %s).'
              % (mask.sum() / len(cloud), mask.sum(), len(cloud), min, max))

    if only_mask:
        return mask

    filtered = cloud[mask]
    return filtered


def filter_grid(cloud, grid_res, keep='first', log=False, rng=default_rng, only_mask=False):
    """Keep single point within each cell. Order is not preserved."""
    assert isinstance(cloud, np.ndarray), type(cloud)
    # assert cloud.dtype.names
    assert isinstance(grid_res, (float, int)) and grid_res > 0.0
    assert keep in ('first', 'random', 'last')

    if cloud.dtype.names:
        cloud = cloud.ravel()
    if keep == 'first':
        pass
    elif keep == 'random':
        rng.shuffle(cloud)
    elif keep == 'last':
        cloud = cloud[::-1]

    x = position(cloud)
    keys = np.floor(x / grid_res).astype(int)
    assert keys.size > 0
    _, ind = np.unique(keys, return_index=True, axis=0)

    if log:
        print('%.3f = %i / %i points kept (grid res. %.3f m).'
              % (len(ind) / len(keys), len(ind), len(keys), grid_res))

    if only_mask:
        return ind

    filtered = cloud[ind]
    return filtered


def filter_cylinder(cloud, radius, axis='z', log=False, only_mask=False):
    """Keep points within cylinder."""
    assert isinstance(cloud, np.ndarray), type(cloud)
    assert isinstance(radius, (float, int)) and radius > 0.0
    assert axis in ('x', 'y', 'z')

    if cloud.dtype.names:
        cloud = cloud.ravel()
    x = position(cloud)
    if axis == 'x':
        mask = np.abs(x[:, 0]) <= radius
    elif axis == 'y':
        mask = np.abs(x[:, 1]) <= radius
    elif axis == 'z':
        mask = np.abs(x[:, 2]) <= radius
    else:
        raise ValueError(axis)

    if log:
        print('%.3f = %i / %i points kept (radius %.3f m).'
              % (mask.sum() / len(cloud), mask.sum(), len(cloud), radius))

    if only_mask:
        return mask

    filtered = cloud[mask]
    return filtered

def valid_point_mask(arr, discard_tf=None, discard_model=None):
    assert isinstance(arr, np.ndarray)
    assert arr.dtype.names
    # Identify valid points, i.e., points with valid depth which are not part
    # of the robot (contained by the discard model).
    # x = position(arr)
    # x = x.reshape((-1, 3)).T
    x = position(arr.ravel()).T
    valid = np.isfinite(x).all(axis=0)
    valid = np.logical_and(valid, (x != 0.0).any(axis=0))
    if discard_tf is not None and discard_model is not None:
        y = affine(discard_tf, x)
        valid = np.logical_and(valid, ~discard_model.contains_point(y))
    return valid.reshape(arr.shape)

def estimate_heightmap(points, d_min=1., d_max=12.8, grid_res=0.1, h_max=1., hm_interp_method='nearest',
                       fill_value=0., robot_z=0., return_filtered_points=False,
                       map_pose=np.eye(4), grass_range=(0.1, 1.0)):
    assert points.ndim == 2
    assert points.shape[1] >= 3  # (N x 3)
    assert len(points) > 0
    assert isinstance(d_min, (float, int)) and d_min >= 0.
    assert isinstance(d_max, (float, int)) and d_max >= 0.
    assert isinstance(grid_res, (float, int)) and grid_res > 0.
    assert isinstance(h_max, (float, int)) and h_max >= 0.
    assert hm_interp_method in ['linear', 'nearest', 'cubic', None]
    assert fill_value is None or isinstance(fill_value, (float, int))
    assert robot_z is None or isinstance(robot_z, (float, int))
    assert isinstance(return_filtered_points, bool)
    assert map_pose.shape == (4, 4)
    assert grass_range is None or isinstance(grass_range, (tuple, list)) and len(grass_range) == 2

    # remove invalid points
    mask_valid = np.isfinite(points).all(axis=1)
    points = points[mask_valid]

    # gravity aligned points
    roll, pitch, yaw = rot2rpy(map_pose[:3, :3])
    R = rpy2rot(roll, pitch, 0.).cpu().numpy()
    points_grav = points @ R.T

    # filter ground (points in a height range from 0 to 0.5 m)
    if grass_range is not None:
        mask_grass = np.logical_and(points[:, 2] >= grass_range[0], points[:, 2] <= grass_range[1])
    else:
        mask_grass = np.zeros(len(points), dtype=bool)

    # height above ground
    mask_h = points_grav[:, 2] <= h_max

    # filter point cloud in a square
    mask_sq = np.logical_and(np.abs(points[:, 0]) <= d_max, np.abs(points[:, 1]) <= d_max)

    # combine and apply masks
    mask = np.logical_and(~mask_grass, mask_h)
    mask = np.logical_and(mask, mask_sq)
    points = points[mask]
    if len(points) == 0:
        if return_filtered_points:
            return None, None
        return None

    # add a point cloud under the robot with robot_z height
    d_robot = d_min if d_min > 0. else 0.6
    n_robot = int(2 * d_robot / grid_res)
    x_robot = np.linspace(-d_robot, d_robot, n_robot)
    y_robot = np.linspace(-d_robot, d_robot, n_robot)
    x_robot, y_robot = np.meshgrid(x_robot, y_robot)
    z_robot = np.full(x_robot.shape, fill_value=robot_z)
    robot_points = np.stack([x_robot, y_robot, z_robot], axis=2)
    robot_points = robot_points.reshape((-1, 3))
    # robot robot points to robot frame
    robot_points = robot_points @ map_pose[:3, :3].T
    points = np.concatenate([points, robot_points], axis=0)

    # create a grid
    n = int(2 * d_max / grid_res)
    xi = np.linspace(-d_max, d_max, n)
    yi = np.linspace(-d_max, d_max, n)
    x_grid, y_grid = np.meshgrid(xi, yi)

    if hm_interp_method is None:
        # estimate heightmap
        z_grid = np.full(x_grid.shape, fill_value=fill_value)
        for i in range(len(points)):
            xp = points[i, 0]
            yp = points[i, 1]
            zp = points[i, 2]
            # find the closest grid point
            idx_x = np.argmin(np.abs(x_grid[0, :] - xp))
            idx_y = np.argmin(np.abs(y_grid[:, 0] - yp))
            # update heightmap
            if z_grid[idx_y, idx_x] == fill_value or zp > z_grid[idx_y, idx_x]:
                z_grid[idx_y, idx_x] = zp
            else:
                # print('Point is lower than the current heightmap value, skipping...')
                pass
        mask_meas = np.asarray(z_grid != fill_value, dtype=np.float32)
    else:
        X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
        z_grid = griddata((X, Y), Z, (xi[None, :], yi[:, None]),
                          method=hm_interp_method, fill_value=fill_value)
        mask_meas = np.full(z_grid.shape, 1., dtype=np.float32)

    z_grid = z_grid.T
    mask_meas = mask_meas.T
    heightmap = {'x': np.asarray(x_grid, dtype=np.float32),
                 'y': np.asarray(y_grid, dtype=np.float32),
                 'z': np.asarray(z_grid, dtype=np.float32),
                 'mask': mask_meas}

    if return_filtered_points:
        return heightmap, points

    return heightmap


def hm_to_cloud(height, cfg, mask=None):
    assert isinstance(height, np.ndarray) or isinstance(height, torch.Tensor)
    assert height.ndim == 2
    if mask is not None:
        assert isinstance(mask, (np.ndarray, torch.Tensor))
        assert mask.ndim == 2
        assert height.shape == mask.shape
        mask = mask.bool() if isinstance(mask, torch.Tensor) else mask.astype(bool)
    z_grid = height
    if isinstance(height, np.ndarray):
        x_grid = np.linspace(-cfg.d_max, cfg.d_max, z_grid.shape[0])
        y_grid = np.linspace(-cfg.d_max, cfg.d_max, z_grid.shape[1])
        x_grid, y_grid = np.meshgrid(x_grid, y_grid)
        hm_cloud = np.stack([x_grid, y_grid, z_grid], axis=2)
    else:
        x_grid = torch.linspace(-cfg.d_max, cfg.d_max, z_grid.shape[0]).to(z_grid.device)
        y_grid = torch.linspace(-cfg.d_max, cfg.d_max, z_grid.shape[1]).to(z_grid.device)
        x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
        hm_cloud = torch.stack([x_grid, y_grid, z_grid], dim=2)
    if mask is not None:
        hm_cloud = hm_cloud[mask]
    hm_cloud = hm_cloud.reshape([-1, 3])
    return hm_cloud


def demo():
    from .datasets import TravData, seq_paths
    from .utils import read_yaml
    from .config import Config

    def show_clouds(points1, points2=None, **kwargs):
        import open3d as o3d

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        pcd1.paint_uniform_color([1.0, 0.0, 0.0])

        if points2 is None:
            o3d.visualization.draw_geometries([pcd1], **kwargs)
            return

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2.paint_uniform_color([0.0, 0.0, 1.0])
        o3d.visualization.draw_geometries([pcd1, pcd2], **kwargs)

    cfg = Config()
    path = '/home/ruslan/workspaces/traversability_ws/src/monoforce'
    cfg.from_yaml(os.path.join(path, 'config/cfg.yaml'))
    cfg.hm_interp_method = None

    lss_cfg = read_yaml(os.path.join(path, 'config/lss.yaml'))
    data_aug_conf = lss_cfg['data_aug_conf']

    # ds = TravData(seq_paths[0], is_train=True, data_aug_conf=data_aug_conf, cfg=cfg)
    # i = 50
    ds = TravData(seq_paths[2], is_train=True, data_aug_conf=data_aug_conf, cfg=cfg)
    i = 5
    cloud = ds.get_cloud(i)
    points = position(cloud)
    map_pose = ds.get_pose(i)
    # map_pose = np.eye(4)

    hm = estimate_heightmap(points, d_min=cfg.d_min, d_max=cfg.d_max, grid_res=cfg.grid_res,
                            h_max=cfg.h_max, hm_interp_method=cfg.hm_interp_method,
                            fill_value=0., robot_z=-0.1, return_filtered_points=False,
                            map_pose=map_pose, grass_range=cfg.grass_range)

    hm_cloud = np.stack([hm['x'], hm['y'], hm['z'].T], axis=2)[hm['mask'].astype(bool).T]
    show_clouds(points, hm_cloud)


if __name__ == '__main__':
    demo()
