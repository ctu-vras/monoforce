import os
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.recfunctions import structured_to_unstructured
from timeit import default_timer as timer
import torch
import yaml
from random import random
from time import sleep


__all__ = [
    'slots',
    'timing',
    'normalize',
    'skew_symmetric',
    'read_yaml',
    'write_to_yaml',
    'str2bool',
    'position',
    'color',
    'load_calib',
    'compile_data',
    'explore_data'
]

def slots(msg):
    """Return message attributes (slots) as list."""
    return [getattr(msg, var) for var in msg.__slots__]


def timing(f):
    def timing_wrapper(*args, **kwargs):
        t0 = timer()
        ret = f(*args, **kwargs)
        t1 = timer()
        print('%s %.6f s' % (f.__name__, t1 - t0))
        # rospy.logdebug('%s %.6f s' % (f.__name__, t1 - t0))
        return ret
    return timing_wrapper


def normalize(x, qlow=0., qhigh=1., eps=1e-6, ):
    assert qlow < qhigh
    assert qlow >= 0 and qhigh <= 1
    assert eps > 0
    """Scale to range 0..1"""
    if isinstance(x, torch.Tensor):
        x_max = torch.quantile(x, qhigh).item()
        x_min = torch.quantile(x, qlow).item()
        x = (x - x_min) / np.max([(x_max - x_min), eps])
        x = x.clamp(0, 1)
    else:
        x_max = np.percentile(x, 100 * qhigh)
        x_min = np.percentile(x, 100 * qlow)
        x = (x - x_min) / np.max([(x_max - x_min), eps])
        x = x.clip(0, 1)
    return x


def skew_symmetric(x):
    U = torch.as_tensor([[0., -x[2], x[1]],
                         [x[2], 0., -x[0]],
                         [-x[1], x[0], 0.]], device=x.device)
    return U


def read_yaml(path):
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


def write_to_yaml(cfg: dict, path):
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

def str2bool(v):
    return v.lower() in ('1', 'yes', 'true', 't', 'y')

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


def load_calib(calib_path):
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
    T = np.asarray(calib['transformations']['T_base_link__base_footprint']['data'], dtype=np.float32).reshape((4, 4))
    calib['clearance'] = np.abs(T[2, 3])

    return calib


def compile_data(val_fraction=0.1, small_data=False, vis=False, Data=None, dphys_cfg=None, lss_cfg=None):
    from torch.utils.data import ConcatDataset, Subset
    from monoforce.datasets import ROUGH, rough_seq_paths
    """
    Compile datasets for LSS model training

    :param lss_cfg: dict, LSS model configuration
    :param dphys_cfg: DPhysConfig, physical robot-terrain interaction configuration
    :param val_fraction: float, fraction of the dataset to use for validation
    :param small_data: bool, debug mode: use small datasets
    :param vis: bool, visualize training samples
    :param kwargs: additional arguments

    :return: train_ds, val_ds
    """
    train_datasets = []
    val_datasets = []
    print('Data paths:', rough_seq_paths)
    if Data is None:
        Data = ROUGH
    for path in rough_seq_paths:
        assert os.path.exists(path)
        train_ds = Data(path, is_train=True, dphys_cfg=dphys_cfg, lss_cfg=lss_cfg)
        val_ds = Data(path, is_train=False, dphys_cfg=dphys_cfg, lss_cfg=lss_cfg)
        if vis:
            explore_data(train_ds)
            vis = False  # visualize only the first dataset sample

        # randomly select a subset of the dataset
        val_ds_size = int(val_fraction * len(train_ds))
        val_ids = np.random.choice(len(train_ds), val_ds_size, replace=False)
        train_ids = np.setdiff1d(np.arange(len(train_ds)), val_ids)
        assert len(train_ids) + len(val_ids) == len(train_ds)
        # check that there is no overlap between train and val ids
        assert len(np.intersect1d(train_ids, val_ids)) == 0

        train_ds = train_ds[train_ids]
        val_ds = val_ds[val_ids]
        print(f'Train dataset from path {path} size is {len(train_ds)}')
        print(f'Validation dataset from path {path} size is {len(val_ds)}')

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    # concatenate datasets
    train_ds = ConcatDataset(train_datasets)
    val_ds = ConcatDataset(val_datasets)

    if small_data:
        train_datasets = [Data(path, is_train=True, dphys_cfg=dphys_cfg, lss_cfg=lss_cfg) for path in rough_seq_paths]
        val_datasets = [Data(path, is_train=False, dphys_cfg=dphys_cfg, lss_cfg=lss_cfg) for path in rough_seq_paths]
        print('Debug mode: using small datasets')
        # concatenate datasets
        train_ds = ConcatDataset(train_datasets)
        val_ds = ConcatDataset(val_datasets)
        # ids = [79]
        ids = np.random.choice(len(train_ds), 16, replace=False).tolist()
        train_ds = Subset(train_ds, ids)
        val_ds = Subset(val_ds, ids)
    print('Concatenated datasets length: train %i, valid: %i' % (len(train_ds), len(val_ds)))

    return train_ds, val_ds


def explore_data(ds, sample_range='random', save=False):
    from tqdm import tqdm
    from monoforce.models.terrain_encoder.lss import LiftSplatShoot
    from monoforce.models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, denormalize_img

    lss_cfg = ds.lss_cfg
    d_max = lss_cfg['grid_conf']['xbound'][1]
    model = LiftSplatShoot(lss_cfg['grid_conf'], lss_cfg['data_aug_conf'], outC=1)

    H, W = ds.lss_cfg['data_aug_conf']['H'], ds.lss_cfg['data_aug_conf']['W']
    cams = ds.camera_names

    if sample_range == 'random':
        sample_range = [np.random.choice(range(len(ds)))]
        print('Selected data sample #{}'.format(sample_range[0]))
    elif sample_range == 'all':
        sample_range = tqdm(range(len(ds)), total=len(ds))
    else:
        assert isinstance(sample_range, list) or isinstance(sample_range, np.ndarray) or isinstance(sample_range, range)

    for sample_i in tqdm(sample_range):
        imgs, rots, trans, intrins, post_rots, post_trans = ds.get_images_data(sample_i)
        height_geom = ds.get_geom_height_map(sample_i)[0]
        height_terrain = ds.get_terrain_height_map(sample_i)[0]
        pts = torch.as_tensor(position(ds.get_cloud(sample_i))).T

        frustum_pts = model.get_geometry(rots[None], trans[None], intrins[None], post_rots[None], post_trans[None]).squeeze(0)

        n_rows, n_cols = 2, int(np.ceil(len(cams) / 2) + 3)
        img_h, img_w = imgs.shape[-2], imgs.shape[-1]
        ratio = img_h / img_w
        fig = plt.figure(figsize=(n_cols * 5, n_rows * 5 * ratio))
        gs = mpl.gridspec.GridSpec(n_rows, n_cols)
        gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

        plt.clf()
        final_ax = plt.subplot(gs[:, -1:])
        for imgi, img in enumerate(imgs):
            cam_pts = ego_to_cam(pts, rots[imgi], trans[imgi], intrins[imgi])
            mask = get_only_in_img_mask(cam_pts, H, W)
            plot_pts = post_rots[imgi].matmul(cam_pts) + post_trans[imgi].unsqueeze(1)

            ax = plt.subplot(gs[imgi // int(np.ceil(len(cams) / 2)), imgi % int(np.ceil(len(cams) / 2))])
            showimg = denormalize_img(img)

            plt.imshow(showimg)
            plt.scatter(plot_pts[0, mask], plot_pts[1, mask], c=pts[2, mask],
                        s=1, alpha=0.4, cmap='jet', vmin=-1., vmax=1.)
            plt.axis('off')
            # camera name as text on image
            plt.text(0.5, 0.9, cams[imgi].replace('_', ' '),
                     horizontalalignment='center', verticalalignment='top',
                     transform=ax.transAxes, fontsize=10)

            plt.sca(final_ax)
            plt.scatter(frustum_pts[imgi, :, :, :, 0].view(-1), frustum_pts[imgi, :, :, :, 1].view(-1),
                        label=cams[imgi].replace('_', ' '), s=0.2, alpha=0.5)

        plt.legend(loc='upper right')
        final_ax.set_aspect('equal')
        plt.title('Frustum points')
        plt.xlim((-d_max, d_max))
        plt.ylim((-d_max, d_max))

        # plot height maps
        plt.subplot(gs[:, -3:-2])
        plt.imshow(height_geom.T, origin='lower', cmap='jet', vmin=-1., vmax=1.)
        # plt.axis('off')
        plt.title('Geom HM')
        # plt.colorbar()

        plt.subplot(gs[:, -2:-1])
        plt.imshow(height_terrain.T, origin='lower', cmap='jet', vmin=-1., vmax=1.)
        # plt.axis('off')
        plt.title('Terrain HM')
        # plt.colorbar()

        if save:
            save_dir = os.path.join(ds.path, 'visuals')
            os.makedirs(save_dir, exist_ok=True)
            imname = f'{ds.ids[sample_i]}.jpg'
            imname = os.path.join(save_dir, imname)
            # print('saving', imname)
            plt.savefig(imname)
            plt.close(fig)
        else:
            plt.show()


class PathLock(object):

    lock_template = '%s.lock'

    def __init__(self, path, interval=1.0, repeat=-1):
        self.path = path
        self.lock_path = PathLock.lock_template % path
        self.locked = False
        self.interval = interval
        self.repeat = repeat

    def sleep(self):
        interval = random() * self.interval
        sleep(interval)

    def lock(self):
        assert not self.locked
        i = -1
        while self.repeat < 0 or i < self.repeat:
            i += 1
            try:
                with open(self.lock_path, 'x'):
                    pass
                self.locked = True
                return self
            except FileExistsError as ex:
                self.sleep()
                continue
        raise PathLockException()

    def unlock(self):
        assert self.locked
        assert os.path.exists(self.lock_path)
        os.remove(self.lock_path)
        self.locked = False

    def __enter__(self):
        return self.lock()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.locked:
            self.unlock()


def write_to_csv(path, text, append=False, create_dirs=True):
    if create_dirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with PathLock(path):
        mode = 'a' if append else 'w'
        with open(path, mode) as f:
            f.write(text)


def append_to_csv(path, text, create_dirs=True):
    write_to_csv(path, text, append=True, create_dirs=create_dirs)


class PathLockException(Exception):
    pass
