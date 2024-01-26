import os.path

import torch
from .geometry import affine
from .transformations import rot2rpy, rpy2rot
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

@timing
def compute_rigid_support(arr, transform=None, range=None, grid=None, scale=1.0, radius=0.1, min_support=30):
    xyz = position(arr)
    xyz = xyz.reshape((-1, 3))

    if transform is not None:
        # xyz = affine(transform, xyz.T).T
        # Only rotate so that range is applied in sensor frame.
        xyz = np.matmul(xyz, transform[:3, :3].T)

    filtered = xyz.copy()
    if range is not None:
        filtered = filter_range(xyz, *range)
    if grid is not None:
        filtered = filter_grid(xyz, grid)

    xyz = scale * xyz
    filtered = scale * filtered

    tree = cKDTree(filtered, compact_nodes=False, balanced_tree=False)
    ind = tree.query_ball_point(xyz, radius, workers=-1)

    support = np.array([len(i) for i in ind]).astype(np.uint32)
    support = support.reshape(arr.shape)

    rigid = support >= min_support

    return support, rigid


def estimate_heightmap(points, d_min=1., d_max=12.8, grid_res=0.1, h_max=1., hm_interp_method='nearest',
                       fill_value=0., robot_z=0., return_filtered_points=False,
                       map_pose=np.eye(4), ground_range=(0., 0.5)):
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
    assert ground_range is None or isinstance(ground_range, tuple) and len(ground_range) == 2

    # remove invalid points
    mask_meas = np.isfinite(points).all(axis=1)
    points = points[mask_meas]

    # gravity aligned points
    roll, pitch, yaw = rot2rpy(map_pose[:3, :3])
    R = rpy2rot(roll, pitch, 0.).cpu().numpy()
    points_grav = points @ R.T

    # filter ground (points in a height range from 0 to 0.5 m)
    if ground_range is not None:
        mask_ground = np.logical_or(points_grav[:, 2] <= ground_range[0], points_grav[:, 2] >= ground_range[1])
        # show_cloud(points)
        # show_cloud(points[mask_ground])
        # show_cloud(points[~mask_ground])
    else:
        mask_ground = np.ones(len(points), dtype=bool)

    # height above ground
    mask_h = points_grav[:, 2] <= h_max

    # filter point cloud in a square
    mask_sq = np.logical_and(np.abs(points[:, 0]) <= d_max, np.abs(points[:, 1]) <= d_max)

    # robot points
    mask_robot = filter_range(points_grav, min=0., max=d_min if d_min > 0. else 0., only_mask=True)

    # set robot points to the minimum height
    if robot_z is None and mask_robot.sum() > 0:
        robot_z = points_grav[mask_robot, 2].min()

    # combine and apply masks
    mask = np.logical_and(mask_ground, mask_h)
    mask = np.logical_and(mask, mask_sq)
    mask = np.logical_and(mask, ~mask_robot)
    points = points[mask]
    if len(points) == 0:
        if return_filtered_points:
            return None, None
        return None

    # # add a point cloud under the robot with robot_z height
    # d_robot = d_min if d_min > 0. else 0.6
    # n_robot = int(2 * d_robot / grid_res)
    # x_robot = np.linspace(-d_robot, d_robot, n_robot)
    # y_robot = np.linspace(-d_robot, d_robot, n_robot)
    # x_robot, y_robot = np.meshgrid(x_robot, y_robot)
    # z_robot = np.full(x_robot.shape, fill_value=robot_z)
    # robot_points = np.stack([x_robot.ravel(), y_robot.ravel(), z_robot.ravel()], axis=1)
    # points = np.concatenate([points, robot_points], axis=0)

    # create a grid
    n = int(2 * d_max / grid_res)
    xi = np.linspace(-d_max, d_max, n)
    yi = np.linspace(-d_max, d_max, n)
    x_grid, y_grid = np.meshgrid(xi, yi)

    if hm_interp_method is None:
        # estimate heightmap
        z_grid = np.full(x_grid.shape, fill_value=fill_value)
        for i in range(len(points)):
            x = points[i, 0]
            y = points[i, 1]
            z = points[i, 2]
            # find the closest grid point
            idx_x = np.argmin(np.abs(x_grid[0, :] - x))
            idx_y = np.argmin(np.abs(y_grid[:, 0] - y))
            # update heightmap
            if z_grid[idx_y, idx_x] == fill_value or z > z_grid[idx_y, idx_x]:
                z_grid[idx_y, idx_x] = z
            else:
                # print('Point is lower than the current heightmap value, skipping...')
                pass
        mask_meas = np.asarray(z_grid != fill_value, dtype=float)
    else:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        z_grid = griddata((x, y), z, (xi[None, :], yi[:, None]),
                          method=hm_interp_method, fill_value=fill_value)
        mask_meas = np.full(z_grid.shape, 1., dtype=float)

    z_grid = z_grid.T
    mask_meas = mask_meas.T
    heightmap = {'x': np.asarray(x_grid, dtype=float),
                 'y': np.asarray(y_grid, dtype=float),
                 'z': np.asarray(z_grid, dtype=float),
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
    import open3d as o3d

    cfg = Config()
    path = '/home/ruslan/workspaces/traversability_ws/src/monoforce'
    cfg.from_yaml(os.path.join(path, 'config/cfg.yaml'))
    cfg.hm_interp_method = None

    lss_cfg = read_yaml(os.path.join(path, 'config/lss.yaml'))
    data_aug_conf = lss_cfg['data_aug_conf']

    ds = TravData(seq_paths[0], is_train=True, data_aug_conf=data_aug_conf, cfg=cfg)

    i = 50
    cloud = ds.get_cloud(i)
    points = position(cloud)
    # map_pose = ds.get_pose(i)
    map_pose = np.eye(4)
    # ground_range = (-0.2, 0.2)
    ground_range = None

    hm = estimate_heightmap(points, d_min=cfg.d_min, d_max=cfg.d_max, grid_res=cfg.grid_res,
                            h_max=cfg.h_max, hm_interp_method=cfg.hm_interp_method,
                            fill_value=0., robot_z=0., return_filtered_points=False,
                            map_pose=map_pose, ground_range=ground_range)

    hm_cloud = np.stack([hm['x'], hm['y'], hm['z'].T], axis=2)[hm['mask'].astype(bool)]

    # visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])

    hm_pcd = o3d.geometry.PointCloud()
    hm_pcd.points = o3d.utility.Vector3dVector(hm_cloud)
    hm_pcd.paint_uniform_color([0.0, 0.0, 1.0])
    o3d.visualization.draw_geometries([pcd, hm_pcd])


if __name__ == '__main__':
    demo()
