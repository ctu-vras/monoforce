import copy
import torch
from PIL import Image
from torch.utils.data import Dataset
from monoforce.cloudproc import estimate_heightmap
from monoforce.models.terrain_encoder.utils import sample_augmentation, img_transform, normalize_img
from monoforce.utils import read_yaml
from monoforce.models.traj_predictor.dphys_config import DPhysConfig
from .utils import (timestamp_to_bag_time,
                    get_ids_2d, get_ids_3d,
                    convert_ts_to_float,
                    get_extrinsics_yaml,
                    get_intrinsics)
from .utils3d import METAINFO
from ..rough import data_dir
import os
import numpy as np
import quaternion
import pandas as pd
from glob import glob
from pathlib import Path


class BaseDataset(Dataset):
    def __init__(self, is_train=False):
        super().__init__()
        self.ids = []
        self.is_train = is_train
        self.camera_names = ['camera_front']
        self.dphys_cfg = DPhysConfig()
        self.lss_cfg = read_yaml(os.path.join(data_dir, '../config', 'lss_cfg_wildscenes.yaml'))
        self.grid_res = self.lss_cfg['grid_conf']['xbound'][2]
        self.calib = {}

    def get_sample(self, i):
        raise NotImplementedError

    def __getitem__(self, i):
        if isinstance(i, (int, np.int64)):
            sample = self.get_sample(i)
            return sample

        ds = copy.deepcopy(self)
        if isinstance(i, (list, tuple, np.ndarray)):
            ds.ids = [self.ids[k] for k in i]
        else:
            assert isinstance(i, (slice, range))
            ds.ids = self.ids[i]
        return ds

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ids)

    def get_cloud(self, i):
        raise NotImplementedError

    def get_cloud_label(self, i):
        raise NotImplementedError

    def get_image(self, i):
        raise NotImplementedError

    def get_geom_height_map(self, i):
        """
        Get height map from lidar point cloud.
        :param i: index of the sample
        :param cached: if True, load height map from file if it exists, otherwise estimate it
        :param dir_name: directory to save/load heightmap
        :return: heightmap (2 x H x W), where 2 is the number of channels (z and mask)
        """
        cloud = torch.as_tensor(self.get_cloud(i))
        heightmap = estimate_heightmap(cloud, d_max=self.dphys_cfg.d_max,
                                      grid_res=self.grid_res,
                                      h_max=self.dphys_cfg.h_max,
                                      r_min=self.dphys_cfg.r_min)
        heightmap = torch.as_tensor(heightmap)
        return heightmap

    def get_terrain_height_map(self, i):
        soft_classes = self.lss_cfg['soft_classes']
        soft_labels = [METAINFO['cidx'][METAINFO['classes'].index(cls)] for cls in soft_classes]
        cloud = self.get_cloud(i)
        label = self.get_cloud_label(i)
        mask = np.isin(label, [soft_labels])
        cloud = cloud[~mask]

        cloud = torch.as_tensor(cloud)
        heightmap = estimate_heightmap(cloud, d_max=self.dphys_cfg.d_max,
                                       grid_res=self.grid_res,
                                       h_max=self.dphys_cfg.h_max,
                                       r_min=self.dphys_cfg.r_min)
        heightmap = torch.as_tensor(heightmap)

        return heightmap

    def get_images_data(self, i):
        imgs = []
        rots = []
        trans = []
        post_rots = []
        post_trans = []
        intrins = []

        for _ in self.camera_names:
            img = self.get_image(i)
            K = self.calib['K']

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = sample_augmentation(self.lss_cfg, is_train=self.is_train)
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                       resize=resize,
                                                       resize_dims=resize_dims,
                                                       crop=crop,
                                                       flip=flip,
                                                       rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # rgb and intrinsics
            img = normalize_img(img)
            K = torch.as_tensor(K)

            # extrinsics
            T_robot_cam = self.calib['E']
            rot = torch.as_tensor(T_robot_cam[:3, :3])
            tran = torch.as_tensor(T_robot_cam[:3, 3])

            imgs.append(img)
            rots.append(rot)
            trans.append(tran)
            intrins.append(K)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        img_data = [torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                    torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans)]
        img_data = [torch.as_tensor(i, dtype=torch.float32) for i in img_data]

        return img_data


class WildScenes(BaseDataset):
    """
    https://csiro-robotics.github.io/WildScenes/
    """

    def __init__(self, seq, is_train=False):
        super().__init__(is_train=is_train)
        self.seq = seq
        self.cloud_paths = sorted(glob(os.path.join(data_dir, f'WildScenes/WildScenes3d/{seq}', 'Clouds', '*')))
        self.cloud_label_paths = sorted(glob(os.path.join(data_dir, f'WildScenes/WildScenes3d/{seq}', 'Labels', '*')))
        self.img_paths = sorted(glob(os.path.join(data_dir, f'WildScenes/WildScenes2d/{seq}', 'image', '*')))
        self.img_label_paths = sorted(glob(os.path.join(data_dir, f'WildScenes/WildScenes2d/{seq}', 'label', '*')))
        _, self.cloud_ts = self.get_cloud_ids()
        self.ids, self.img_ts = self.get_image_ids()
        self.calib = self.get_calib()
        self.poses = self.get_poses()

    def get_cloud_ids(self):
        cloud_seq_path = os.path.join(data_dir, f'WildScenes/WildScenes3d/{self.seq}')
        cloud_timestamp_strings = [
            timestamp_to_bag_time(t) for t in get_ids_3d(Path(cloud_seq_path))
        ]
        cloud_timestamp_strings = sorted(cloud_timestamp_strings)
        cloud_timestamps = convert_ts_to_float(cloud_timestamp_strings)
        return cloud_timestamp_strings, cloud_timestamps

    def get_image_ids(self):
        image_seq_path = os.path.join(data_dir, f'WildScenes/WildScenes2d/{self.seq}')
        image_timestamp_strings = [
            timestamp_to_bag_time(t) for t in get_ids_2d(Path(image_seq_path))
        ]
        image_timestamp_strings = sorted(image_timestamp_strings)
        image_timestamps = convert_ts_to_float(image_timestamp_strings)
        return image_timestamp_strings, image_timestamps

    def get_cloud(self, i):
        img_i = self.get_cloud_i_closest_to_image(i)
        cloud_path = self.cloud_paths[img_i]
        cloud = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 3)
        return cloud

    def get_cloud_label(self, i):
        img_i = self.get_cloud_i_closest_to_image(i)
        cloud_label_path = self.cloud_label_paths[img_i]
        cloud_label = np.fromfile(cloud_label_path, dtype=np.int32)
        return cloud_label

    def get_image(self, i):
        img_path = self.img_paths[i]
        img = Image.open(img_path)
        return img

    def get_cloud_i_closest_to_image(self, i):
        img_ts = self.img_ts[i]
        cloud_ts = np.asarray(self.cloud_ts)
        img_ts = np.asarray(img_ts)
        diff = np.abs(cloud_ts - img_ts)
        bestcloud_i = np.argmin(diff)
        bestdiff = diff[bestcloud_i]
        if bestdiff > 5:
            print(f"Warning: For this image idx there is no suitable point cloud anywhere near this image")
            # raise ValueError("For this image idx there is no suitable point cloud anywhere near this image")
        return bestcloud_i

    def get_calib(self):
        path = os.path.join(data_dir, f'WildScenes/WildScenes2d/{self.seq}')
        params = read_yaml(os.path.join(path, 'camera_calibration.yaml'))['centre-camera']
        K, D = get_intrinsics(params['intrinsics'])
        E = get_extrinsics_yaml(params['extrinsics'])
        calib = {'K': K,
                 'D': D,
                 'E': E,
                 'imgW': 2016, 'imgH': 1512}
        return calib

    def get_poses(self):
        image_seq_path = os.path.join(data_dir, f'WildScenes/WildScenes2d/{self.seq}')
        df = pd.read_csv(os.path.join(image_seq_path, 'poses2d.csv'), sep=' ').sort_index()
        xyz_q_cam = np.array(df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']])
        R_cam = quaternion.as_rotation_matrix(quaternion.from_float_array(xyz_q_cam[:, 3:]))
        poses_cam = np.zeros((len(xyz_q_cam), 4, 4))
        poses_cam[:, :3, :3] = R_cam
        poses_cam[:, :3, 3] = xyz_q_cam[:, :3]
        poses_cam[:, 3, 3] = 1.0

        E = self.calib['E']
        poses = poses_cam @ E.T

        return poses

    def get_pose(self, i):
        return self.poses[i]

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_images_data(i)
        hm_geom = self.get_geom_height_map(i)
        hm_terrain = self.get_terrain_height_map(i)
        return (imgs, rots, trans, intrins, post_rots, post_trans,
                hm_geom, hm_terrain)


def main():
    import open3d as o3d
    from .utils3d import METAINFO
    from monoforce.utils import explore_data
    from tqdm import tqdm
    import matplotlib as mpl
    mpl.use('TkAgg')

    # seq = np.random.choice(['K-01', 'K-03', 'V-01', 'V-02', 'V-03'])
    seq = 'K-01'
    ds = WildScenes(seq=seq, is_train=False)

    # i = np.random.randint(len(ds))
    i = 200
    explore_data(ds, sample_range=[i])
    cloud = ds.get_cloud(i)
    label = ds.get_cloud_label(i)
    print(cloud.shape, label.shape)

    # remove points belonging to soft classes from the cloud
    soft_classes = ds.lss_cfg['soft_classes']
    soft_labels = [METAINFO['cidx'][METAINFO['classes'].index(cls)] for cls in soft_classes]
    mask = np.isin(label, [soft_labels])
    cloud = cloud[~mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
