from .geometry import affine
from .utils import timing
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.spatial import cKDTree
from scipy.interpolate import griddata

default_rng = np.random.default_rng(135)


def keep_mask(n, indices):
    mask = np.zeros(n, dtype=bool)
    mask[indices] = 1
    return mask


def remove_mask(n, indices):
    mask = np.ones(n, dtype=bool)
    mask[indices] = 0
    return mask

def position(cloud):
    """Cloud to point positions (xyz)."""
    if cloud.dtype.names:
        x = structured_to_unstructured(cloud[['x', 'y', 'z']])
    else:
        x = cloud
    return x

def color(cloud):
    """Color to rgb."""
    if cloud.dtype.names:
        rgb = structured_to_unstructured(cloud[['r', 'g', 'b']])
    else:
        rgb = cloud
    return rgb


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


def filter_grid(cloud, grid_res, keep='first', log=False, rng=default_rng, return_mask=False):
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

    filtered = cloud[ind]
    if return_mask:
        return filtered, ind

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


def estimate_heightmap(points, d_min=1., d_max=12.8, grid_res=0.1, h_max=0., hm_interp_method='nearest',
                       fill_value=0., robot_z=0., return_filtered_points=False):
    assert points.ndim == 2
    assert points.shape[1] >= 3  # (N x 3)
    assert len(points) > 0
    assert isinstance(d_min, (float, int)) and d_min >= 0.
    assert isinstance(d_max, (float, int)) and d_max >= 0.
    assert isinstance(grid_res, (float, int)) and grid_res > 0.
    assert isinstance(h_max, (float, int)) and h_max >= 0.
    assert hm_interp_method in ['linear', 'nearest', 'cubic', None]
    assert fill_value is None or isinstance(fill_value, (float, int))

    # remove invalid points
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]

    # filter height outliers points
    z = points[:, 2]
    h_min_out = z[z > np.percentile(z, 2)].min()
    h_max_out = z[z < np.percentile(z, 98)].max()
    points = points[points[:, 2] > h_min_out]
    points = points[points[:, 2] < h_max_out]

    # height above ground
    points = points[points[:, 2] < h_max]
    if len(points) == 0:
        if return_filtered_points:
            return None, None
        return None

    # filter point cloud in a square
    mask_x = np.logical_and(points[:, 0] >= -d_max, points[:, 0] <= d_max)
    mask_y = np.logical_and(points[:, 1] >= -d_max, points[:, 1] <= d_max)
    mask = np.logical_and(mask_x, mask_y)
    points = points[mask]

    if len(points) == 0:
        if return_filtered_points:
            return None, None
        return None

    # robot points
    robot_mask = filter_range(points, min=0., max=d_min if d_min > 0. else 0., only_mask=True)

    # set robot points to the minimum height
    if robot_z is None and robot_mask.sum() > 0:
        robot_z = points[robot_mask, 2].min()

    # remove robot points
    if robot_mask.sum() > 0:
        points = points[~robot_mask]

    # add a point cloud under the robot with robot_z height
    d_robot = d_min if d_min > 0. else 0.6
    n_robot = int(2 * d_robot / grid_res)
    x_robot = np.linspace(-d_robot, d_robot, n_robot)
    y_robot = np.linspace(-d_robot, d_robot, n_robot)
    x_robot, y_robot = np.meshgrid(x_robot, y_robot)
    z_robot = np.full(x_robot.shape, fill_value=robot_z)
    robot_points = np.stack([x_robot.ravel(), y_robot.ravel(), z_robot.ravel()], axis=1)
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
        mask = np.asarray(z_grid != fill_value, dtype=float)
    else:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        z_grid = griddata((x, y), z, (xi[None, :], yi[:, None]),
                          method=hm_interp_method, fill_value=fill_value)
        mask = np.full(z_grid.shape, 1., dtype=float)

    z_grid = z_grid.T
    heightmap = {'x': np.asarray(x_grid, dtype=float),
                 'y': np.asarray(y_grid, dtype=float),
                 'z': np.asarray(z_grid, dtype=float),
                 'mask': mask}

    if return_filtered_points:
        return heightmap, points

    return heightmap
