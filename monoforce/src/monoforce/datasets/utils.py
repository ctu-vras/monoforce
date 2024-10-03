import os
import numpy as np
from ..dphys_config import DPhysConfig
from ..transformations import xyz_rpy_to_matrix
from scipy.spatial.transform import Rotation


__all__ = [
    'get_robingas_data',
    'get_kkt_data',
    'get_simple_data'
]


def get_robingas_data(cfg: DPhysConfig(),
                      path='/home/ruslan/data/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34/',
                      i=None):
    from monoforce.datasets.robingas import RobinGasBase
    # Load traversability data
    assert os.path.exists(path)
    ds = RobinGasBase(path, dphys_cfg=cfg)
    i = np.random.choice(range(len(ds))) if i is None else i
    print('Selected data sample #{}'.format(i))
    sample = ds[i]
    cloud, traj, height = sample
    traj['stamps'] = traj['stamps'] - traj['stamps'][0]
    return height, traj


def get_kkt_data(cfg: DPhysConfig(), i=None,
                 path='/home/ruslan/workspaces/traversability_ws/src/pose-consistency-KKT-loss/', dt=3.):
    import sys
    sys.path.append(os.path.join(path, 'scripts'))
    import dataset_real_rpz

    # non-configurable parameters (constants) of the dataset
    assert cfg.grid_res == 0.1
    assert cfg.d_min == -12.75
    assert cfg.d_max == 12.85

    dataset = dataset_real_rpz.Dataset(os.path.join(path, 'data', 's2d_tst/'))

    i = np.random.choice(range(len(dataset))) if i is None else i
    print('Selected data sample #{}'.format(i))
    data = dataset[i]

    # get height map
    height = data['label_dem'].squeeze()

    # get trajectory
    roll_label = data['label_rpz'][0].T
    pitch_label = data['label_rpz'][1].T
    z_label = data['label_rpz'][2].T
    yaw_label = data['yaw'].T
    xy_grid = 1. - np.isnan(z_label)
    # find non zero elements ids of the 2D array
    x_grid, y_grid = np.where(xy_grid)
    z = z_label[x_grid, y_grid]
    roll = roll_label[x_grid, y_grid]
    pitch = pitch_label[x_grid, y_grid]
    yaw = yaw_label[x_grid, y_grid]
    x = x_grid * cfg.grid_res + cfg.d_min
    y = y_grid * cfg.grid_res + cfg.d_min
    xyz_rpy = np.vstack((x, y, z, roll, pitch, yaw)).T

    # transform xyz_rpy to transformation matrices (N x 4 x 4)
    poses = np.asarray([xyz_rpy_to_matrix(p) for p in xyz_rpy])

    traj = {'poses': poses, 'stamps': np.arange(poses.shape[0]) * dt}
    height = height.T

    return height, traj


def get_simple_data(cfg: DPhysConfig):
    h, w = (cfg.d_max - cfg.d_min) / cfg.grid_res, (cfg.d_max - cfg.d_min) / cfg.grid_res
    h, w = int(h), int(w)
    height = np.zeros((h, w))
    poses = np.asarray([np.eye(4) for _ in range(10)])
    N = 10
    poses[:, 0, 3] = np.linspace(0, 0.6 * cfg.d_max, N)
    poses[:, 1, 3] = np.linspace(0, 0.2 * cfg.d_max, N) ** 1.
    Rs = [Rotation.from_euler('z', np.arctan((poses[i+1, 1, 3] - poses[i, 1, 3]) / (poses[i+1, 0, 3] - poses[i, 0, 3]))).as_matrix() for i in range(N-1)]
    Rs.append(Rs[-1])
    poses[:, :3, :3] = np.asarray(Rs)
    stamps = np.arange(N)
    traj = {'poses': poses, 'stamps': stamps}
    return height, traj


