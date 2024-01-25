import os
import numpy as np
import yaml

from ..config import Config
from ..transformations import xyz_rpy_to_matrix
from ..vis import set_axes_equal
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D


__all__ = [
    'get_robingas_data',
    'get_kkt_data',
    'get_simple_data',
    'visualize_data'
]


def get_robingas_data(cfg: Config(),
                      path='/home/ruslan/data/bags/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/',
                      i=None):
    from monoforce.datasets.data import DEMPathData
    # Load traversability data
    assert os.path.exists(path)
    ds = DEMPathData(path, cfg=cfg)
    i = np.random.choice(range(len(ds))) if i is None else i
    print('Selected data sample #{}'.format(i))
    sample = ds[i]
    cloud, traj, height = sample
    traj['stamps'] = traj['stamps'] - traj['stamps'][0]
    return height, traj


def get_kkt_data(cfg: Config(), i=None,
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


def get_simple_data(cfg: Config):
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


def visualize_data(heightmap, traj, img=None, cfg=Config()):
    height = heightmap['z']

    plt.figure(figsize=(20, 10))
    # add subplot
    ax = plt.subplot(131)
    ax.set_title('Height map')
    ax.imshow(height)
    x, y = traj['poses'][:, 0, 3], traj['poses'][:, 1, 3]
    h, w = height.shape
    x_grid, y_grid = x / cfg.grid_res + w / 2, y / cfg.grid_res + h / 2
    plt.plot(y_grid, x_grid, 'rx-', label='Robot trajectory')
    time_ids = np.linspace(1, len(x_grid), len(x_grid), dtype=int)
    # plot time indices for each waypoint in a trajectory
    for i, txt in enumerate(time_ids):
        ax.annotate(txt, (y_grid[i], x_grid[i]))
    plt.legend()

    # visualize heightmap as a surface in 3D
    ax = plt.subplot(132, projection='3d')
    ax.plot_surface(heightmap['x'], heightmap['y'], heightmap['z'], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Height map')
    set_axes_equal(ax)

    if img is not None:
        # visualize image
        ax = plt.subplot(133)
        ax.imshow(img)
        ax.set_title('Camera view')
        ax.axis('off')

    plt.show()


def load_cam_calib(calib_path):
    calib = {}
    # read camera calibration
    cams_path = os.path.join(calib_path, 'cameras')
    if not os.path.exists(cams_path):
        print('No cameras calibration found in path {}'.format(cams_path))
        return None

    for file in os.listdir(cams_path):
        if file.endswith('.yaml'):
            with open(os.path.join(cams_path, file), 'r') as f:
                cam_info = yaml.load(f, Loader=yaml.FullLoader)
                calib[file.replace('.yaml', '')] = cam_info
            f.close()
    # read cameras-lidar transformations
    trans_path = os.path.join(calib_path, 'transformations.yaml')
    with open(trans_path, 'r') as f:
        transforms = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    calib['transformations'] = transforms

    return calib


if __name__ == '__main__':
    cfg = Config()
    # cfg.d_min, cfg.d_max = -12.75, 12.85
    # cfg.grid_res = 0.1

    cfg.d_min, cfg.d_max = -8., 8.
    cfg.grid_res = 0.2

    height, traj = get_robingas_data(cfg)
    visualize_data(height, traj, cfg)
    # height, traj = get_kkt_data(cfg)
    # visualize_data(height, traj, cfg)
    # height, traj = get_simple_data(cfg)
    # visualize_data(height, traj, cfg)
