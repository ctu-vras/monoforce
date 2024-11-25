import torch
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import numpy as np

default_rng = np.random.default_rng(135)


__all__ = [
    'filter_range',
    'filter_grid',
    'filter_cylinder',
    'filter_box',
    'estimate_heightmap',
    'hm_to_cloud',
    'within_bounds',
    'points2range_img',
    'merge_heightmaps',
]

def position(cloud):
    """Cloud to point positions (xyz)."""
    if cloud.dtype.names:
        x = structured_to_unstructured(cloud[['x', 'y', 'z']])
    else:
        x = cloud
    return x


def within_bounds(x, min=None, max=None, bounds=None, log_variable=None):
    """Mask of x being within bounds  min <= x <= max."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    assert isinstance(x, torch.Tensor)

    keep = torch.ones((x.numel(),), dtype=torch.bool, device=x.device)

    if bounds:
        assert min is None and max is None
        min, max = bounds

    if min is not None and min > -float('inf'):
        if not isinstance(min, torch.Tensor):
            min = torch.tensor(min)
        keep = keep & (x.flatten() >= min)
    if max is not None and max < float('inf'):
        if not isinstance(max, torch.Tensor):
            max = torch.tensor(max)
        keep = keep & (x.flatten() <= max)

    if log_variable is not None:
        print('%.3f = %i / %i points kept (%.3g <= %s <= %.3g).'
              % (keep.double().mean(), keep.sum(), keep.numel(),
                 min if min is not None else float('nan'),
                 log_variable,
                 max if max is not None else float('nan')))

    return keep


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


def filter_box(cloud, box_size, box_pose=None, only_mask=False):
    """Keep points with rectangular bounds."""
    assert isinstance(cloud, np.ndarray)
    assert isinstance(box_size, (tuple, list)) and len(box_size) == 3
    assert all(isinstance(s, (float, int)) and s > 0.0 for s in box_size)
    assert box_pose is None or isinstance(box_pose, np.ndarray)

    if cloud.dtype.names:
        pts = position(cloud)
    else:
        pts = cloud
    assert pts.ndim == 2, "Input points tensor dimensions is %i (only 2 is supported)" % pts.ndim
    pts = torch.from_numpy(pts)

    if box_pose is None:
        box_pose = np.eye(4)
    assert isinstance(box_pose, np.ndarray)
    assert box_pose.shape == (4, 4)
    box_center = box_pose[:3, 3]
    box_orient = box_pose[:3, :3]

    pts = (pts - box_center) @ box_orient

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    keep_x = within_bounds(x, min=-box_size[0] / 2, max=+box_size[0] / 2)
    keep_y = within_bounds(y, min=-box_size[1] / 2, max=+box_size[1] / 2)
    keep_z = within_bounds(z, min=-box_size[2] / 2, max=+box_size[2] / 2)

    keep = torch.logical_and(keep_x, keep_y)
    keep = torch.logical_and(keep, keep_z)

    if only_mask:
        return keep
    filtered = cloud[keep]
    return filtered


def estimate_heightmap(points, grid_res, d_max, h_max, r_min=None, map_pose=None):
    # remove nans from the point cloud if any
    mask = ~torch.isnan(points).any(dim=1)
    points = points[mask]

    if map_pose is not None:
        # move to gravity-aligned frame
        assert map_pose.shape == (4, 4)
        roll, pitch, yaw = Rotation.from_matrix(map_pose[:3, :3]).as_euler('xyz')
        R = Rotation.from_euler('xyz', [roll, pitch, 0.]).as_matrix()
        R = torch.as_tensor(R, dtype=torch.float32)
        points = points @ R.T

    if r_min is not None:
        # remove points in a r_min radius
        distances = torch.norm(points[:, :2], dim=1)
        mask = distances > r_min
        points = points[mask]

    mask = ((points[:, 0] > -d_max) & (points[:, 0] < d_max) &
            (points[:, 1] > -d_max) & (points[:, 1] < d_max) &
            (points[:, 2] > -h_max) & (points[:, 2] < h_max))
    points = points[mask]

    # Extract X, Y, Z
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Compute grid dimensions
    x_bins = torch.arange(-d_max, d_max, grid_res)
    y_bins = torch.arange(-d_max, d_max, grid_res)

    # Digitize coordinates to find grid indices
    x_indices = torch.bucketize(x.contiguous(), x_bins) - 1
    y_indices = torch.bucketize(y.contiguous(), y_bins) - 1

    # Use scatter_reduce to populate the heightmap
    flat_indices = y_indices * len(x_bins) + x_indices  # Flattened indices
    flat_heightmap = torch.full((len(y_bins) * len(x_bins),), float('nan'))

    # Use scatter_reduce to take the maximum height per grid cell
    flat_heightmap = torch.scatter_reduce(
        flat_heightmap,
        dim=0,
        index=flat_indices,
        src=z,
        reduce="amax",
        include_self=False
    )

    # Reshape back to 2D
    heightmap = flat_heightmap.view(len(y_bins), len(x_bins))

    # Replace NaNs with a default value (e.g., 0.0)
    measurements_mask = ~torch.isnan(heightmap)
    # heightmap = torch.nan_to_num(heightmap, nan=0.0)
    heightmap = torch.nan_to_num(heightmap, nan=-h_max)

    # TODO: fix that bug, not sure why need to do that
    heightmap = heightmap.T
    measurements_mask = measurements_mask.T

    hm = torch.stack([heightmap, measurements_mask], dim=0)  # (2, H, W)

    return hm


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

def points2range_img(x, y, z,
                     H=None, W=None,
                     fov_up=None, fov_down=None, fov_left=None, fov_right=None, ang_res=np.pi/180.):
    """
    Convert 3D points to depth image.

    @param x: x-coordinates of the points.
    @param y: y-coordinates of the points.
    @param z: z-coordinates of the points.
    @param H: height of the depth image.
    @param W: width of the depth image.
    @param fov_up: upper bound of the vertical field of view.
    @param fov_down: lower bound of the vertical field of view.
    @param fov_left: left bound of the horizontal field of view.
    @param fov_right: right bound of the horizontal field of view.
    @param ang_res: angular resolution of the depth image.
    @return: depth image.

    Example:
    ```
    points = np.random.rand(100, 3)
    depth_img = points2depth_img(points)
    ```
    If you want to specify the size of the depth image, you can do:
    ```
    depth_img = points2depth_img(points, H=64, W=256)
    ```
    If FOV is not specified, it will be computed from the points.
    You can specify the FOV or the shape of the depth image.
    """
    assert len(x) == len(y) == len(z)

    points = np.stack([x, y, z], axis=1)
    depth = np.linalg.norm(points, axis=1)
    elev = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    azim = -np.arctan2(y, x)

    if fov_up is None:
        fov_up = elev.max()
    if fov_down is None:
        fov_down = elev.min()
    if fov_left is None:
        fov_left = azim.min()
    if fov_right is None:
        fov_right = azim.max()

    if H is None or W is None:
        n_bins = int((fov_up - fov_down) / ang_res)
        n_cols = int((fov_right - fov_left) / ang_res)
    else:
        n_bins, n_cols = H, W

    depth_img = np.zeros((n_bins, n_cols))
    azim_bins = np.digitize(azim, np.linspace(fov_left, fov_right, n_cols)) - 1  # 0-based
    elev_bins = np.digitize(elev, np.linspace(fov_down, fov_up, n_bins)) - 1

    depth_img[elev_bins, azim_bins] = depth

    return depth_img


def merge_heightmaps(new_points, prev_points, grid_res=None):
    """
    Ones new cloud is received, find the overlapping region with the existing cloud and merge them
    """
    assert new_points.ndim == 2 and new_points.shape[1] >= 3, 'Invalid cloud shape %s' % new_points.shape
    assert prev_points.ndim == 2 and prev_points.shape[1] >= 3, 'Invalid cloud shape %s' % prev_points.shape
    if grid_res is not None:
        assert isinstance(grid_res, (float, int)) and grid_res > 0.
    else:
        grid_res = np.mean([np.mean(np.diff(np.unique(new_points[:, 0]))),
                            np.mean(np.diff(np.unique(new_points[:, 1])))])
        print('Grid resolution not provided, using estimated %.3f m' % grid_res)

    # find overlapping region
    tree = cKDTree(prev_points[:, :2])
    dists, idxs = tree.query(new_points[:, :2], k=1)
    common_points_mask = dists < grid_res
    if not np.any(common_points_mask):
        print('No common points found')
        return prev_points

    X = new_points[:, 0]
    Y = new_points[:, 1]
    Z = new_points[:, 2]
    # update heightmap
    Z_prev = prev_points[idxs[common_points_mask], 2]
    Z[common_points_mask] = np.mean([Z_prev, Z[common_points_mask]], axis=0)
    assert len(X) == len(Y) == len(Z), 'Invalid cloud shape'

    # update current points
    prev_points = np.column_stack((X, Y, Z))

    return prev_points
