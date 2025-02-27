import copy
import torch
from PIL import Image
from torch.utils.data import Dataset
from monoforce.cloudproc import estimate_heightmap
from monoforce.models.terrain_encoder.utils import sample_augmentation, img_transform, normalize_img
from monoforce.utils import read_yaml
from monoforce.models.traj_predictor.dphys_config import DPhysConfig
from .utils import get_extrinsics, get_intrinsics
from .utils3d import METAINFO
import os
import numpy as np
import pandas as pd
from glob import glob

monoforce_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.realpath(os.path.join(monoforce_dir, 'data'))


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
                                       h_max=self.dphys_cfg.h_max - 1.2,
                                       h_min=-self.dphys_cfg.h_max - 1.2,
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
                                       h_max=self.dphys_cfg.h_max - 1.2,
                                       h_min=-self.dphys_cfg.h_max - 1.2,
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
        self.cloud_ids = self.get_cloud_ids()
        self.img_ids = self.get_image_ids()
        self.ids = self.img_ids
        self.calib = self.get_calib()

    def get_cloud_ids(self):
        image_seq_path = os.path.join(data_dir, f'WildScenes/WildScenes2d/{self.seq}')
        df = pd.read_csv(os.path.join(image_seq_path, 'corresponding_cloud_id.csv'))
        cloud_ids = df['cloud_i'].values
        return cloud_ids

    def get_image_ids(self):
        image_seq_path = os.path.join(data_dir, f'WildScenes/WildScenes2d/{self.seq}')
        df = pd.read_csv(os.path.join(image_seq_path, 'corresponding_cloud_id.csv'))
        img_ids = df['img_i'].values
        return img_ids

    def get_cloud(self, i):
        cloud_i = self.cloud_ids[i]
        cloud_path = self.cloud_paths[cloud_i]
        cloud = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 3)
        return cloud

    def get_cloud_label(self, i):
        cloud_i = self.cloud_ids[i]
        cloud_label_path = self.cloud_label_paths[cloud_i]
        cloud_label = np.fromfile(cloud_label_path, dtype=np.int32)
        return cloud_label

    def get_image(self, i):
        img_i = self.img_ids[i]
        img_path = self.img_paths[img_i]
        img = Image.open(img_path)
        return img

    def get_calib(self):
        path = os.path.join(data_dir, f'WildScenes/WildScenes2d/{self.seq}')
        params = read_yaml(os.path.join(path, 'camera_calibration.yaml'))['centre-camera']
        K, D = get_intrinsics(params['intrinsics'])
        E = get_extrinsics(params['extrinsics'])
        calib = {'K': K,
                 'D': D,
                 'E': E,
                 'imgW': 2016, 'imgH': 1512}
        return calib

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_images_data(i)
        hm_geom = self.get_geom_height_map(i)
        hm_terrain = self.get_terrain_height_map(i)
        return (imgs, rots, trans, intrins, post_rots, post_trans,
                hm_geom, hm_terrain)


def show_segmented_cloud_and_heightmap():
    import open3d as o3d
    from .utils3d import METAINFO
    from monoforce.utils import explore_data
    from tqdm import tqdm
    import matplotlib as mpl
    mpl.use('TkAgg')

    seq = np.random.choice(['K-01', 'K-03', 'V-01', 'V-02', 'V-03'])
    # seq = 'K-01'
    ds = WildScenes(seq=seq, is_train=False)

    i = np.random.randint(len(ds))
    # i = 200
    explore_data(ds, sample_range=[i])
    cloud = ds.get_cloud(i)
    label = ds.get_cloud_label(i)
    print(cloud.shape, label.shape)
    class_2_rgb = {c: p for c, p in zip(METAINFO['classes'], METAINFO['palette'])}
    color = np.array([class_2_rgb[METAINFO['classes'][cidx]] for cidx in label])

    # remove points belonging to soft classes from the cloud
    soft_classes = ds.lss_cfg['soft_classes']
    soft_labels = [METAINFO['cidx'][METAINFO['classes'].index(cls)] for cls in soft_classes]
    soft_mask = np.isin(label, [soft_labels])

    hm = ds.get_terrain_height_map(i).numpy()
    z_grid, mask = hm[0], np.asarray(hm[1], dtype=bool)
    x_grid = ds.dphys_cfg.x_grid.numpy()
    y_grid = ds.dphys_cfg.y_grid.numpy()
    hm_points = np.stack([x_grid, y_grid, z_grid], axis=2)[mask].reshape(-1, 3)
    pcd_hm = o3d.geometry.PointCloud()
    pcd_hm.points = o3d.utility.Vector3dVector(hm_points)
    pcd_hm.paint_uniform_color([0.5, 0.5, 0.5])

    cloud = cloud[~soft_mask]
    color = color[~soft_mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.colors = o3d.utility.Vector3dVector(color / 255.)

    # coordinates of the camera
    E = ds.calib['E']
    robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    robot_frame.transform(np.eye(4))
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    camera_frame.transform(E)
    ground_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    T = np.eye(4)
    T[2, 3] = -1.2
    ground_frame.transform(T)

    o3d.visualization.draw_geometries([pcd, camera_frame, robot_frame, ground_frame])


def main():
    show_segmented_cloud_and_heightmap()


if __name__ == "__main__":
    main()
