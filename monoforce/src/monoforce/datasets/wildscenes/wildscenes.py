import copy
import torch
from PIL import Image
from torch.utils.data import Dataset
from monoforce.models.terrain_encoder.utils import sample_augmentation, img_transform, normalize_img
from monoforce.utils import read_yaml
from .utils3d import load_pcd
from .utils import (timestamp_to_bag_time,
                    get_ids_2d, get_ids_3d,
                    convert_ts_to_float,
                    get_extrinsics_yaml,
                    get_intrinsics,
                    viz_image,
                    read_rgb_image)
from ..rough import data_dir
import os
import open3d as o3d
import numpy as np
import quaternion
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from pathlib import Path


class BaseDataset(Dataset):
    def __init__(self, is_train=False):
        super().__init__()
        self.ids = []
        self.is_train = is_train

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
        self.ids, self.cloud_ts = self.get_cloud_ids()
        _, self.img_ts = self.get_image_ids()
        self.calib = self.get_calib()
        self.camera_names = ['camera_front']
        self.lss_cfg = read_yaml(os.path.join(data_dir, '../config', 'lss_cfg_wildscenes.yaml'))

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
        cloud_path = self.cloud_paths[i]
        cloud = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 3)
        return cloud

    def get_image(self, i):
        img_path = self.img_paths[i]
        img = Image.open(img_path)
        return img

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

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_images_data(i)
        traj_ts, states = self.get_states_traj(i)
        Xs, Xds, Rs, Omegas = states
        hm_geom = self.get_geom_height_map(i)
        hm_terrain = self.get_terrain_height_map(i)
        pose0 = torch.as_tensor(self.get_initial_pose_on_heightmap(i), dtype=torch.float32)
        return (imgs, rots, trans, intrins, post_rots, post_trans,
                hm_geom, hm_terrain,
                pose0,
                traj_ts, Xs, Xds, Rs, Omegas)


def project_labelcloud_to_image(
    cloud_wrt_camera, cameraparams, rgb_img, label_img, image_name, vizoutfolder=None
):
    # assumes the points are in the local coordinate frame of the camera, not the global coordinate frame

    intrinsics_K, intrinsics_D = get_intrinsics(
        cameraparams["centre-camera"]["intrinsics"]
    )
    imgpoints, _ = cv2.projectPoints(
        cloud_wrt_camera[:, :3].astype(np.float64),
        np.zeros((1, 3)),
        np.zeros((1, 3)),
        intrinsics_K,
        np.zeros(5),
    )

    imgW, imgH = 2016, 1512

    depth_map = np.ones((imgW, imgH)) * np.inf
    dcount = 0
    for p_id, p in enumerate(imgpoints):
        # x, y = int(p[0][0]), int(p[0][1])
        x, y = int(imgpoints[p_id][0][0]), int(imgpoints[p_id][0][1])
        d = cloud_wrt_camera[p_id][2]
        if (x < 0) or (x >= imgW) or (y < 0) or (y >= imgH) or (d <= 1):  # or (d >= 30)
            continue

        # d = point_depths[p_id]
        if d >= depth_map[x, y]:
            continue
        depth_map[x, y] = d
        dcount += 1
        drawcolor = (255, 0, 0)

        rgb_img = cv2.circle(rgb_img, (x, y), radius=2, color=drawcolor, thickness=-1)

    plt.imshow(rgb_img)

    if vizoutfolder is not None:
        viz_image(rgb_img, vizpath=os.path.join(vizoutfolder, "lidarproject_" + image_name))

    return dcount

def project_cloud_to_image():
    import matplotlib as mpl
    mpl.use('TkAgg')

    loaddir = os.path.join(data_dir, "WildScenes/WildScenes3d/K-01")

    cloud_xyz = sorted(glob(os.path.join(loaddir, 'Clouds', '*')))
    labels = sorted(glob(os.path.join(loaddir, 'Labels', '*')))

    twopath = loaddir.replace('WildScenes3d', 'WildScenes2d')
    images = sorted(glob(os.path.join(twopath, 'image', '*')))
    labelimages = sorted(glob(os.path.join(twopath, 'label', '*')))

    image_timestamp_strings = [
        timestamp_to_bag_time(t) for t in get_ids_2d(Path(twopath))
    ]
    cloud_timestamp_strings = [
        timestamp_to_bag_time(t) for t in get_ids_3d(Path(loaddir))
    ]
    image_timestamp_strings = sorted(image_timestamp_strings)
    cloud_timestamp_strings = sorted(cloud_timestamp_strings)

    image_timestamps = convert_ts_to_float(image_timestamp_strings)
    cloud_timestamps = convert_ts_to_float(cloud_timestamp_strings)

    idx = 200  # temp
    thistimestamp = image_timestamps[idx]

    # search for nearest cloud for this image:
    bestdiff = 1e10
    bestcloudidx = -1
    for cloudidx, cloudts in enumerate(cloud_timestamps):
        tdiff = np.abs(thistimestamp - cloudts)
        if tdiff < bestdiff:
            bestdiff = tdiff
            bestcloudidx = cloudidx
    if bestdiff > 5:
        raise ValueError("For this image idx there is no suitable point cloud anywhere near this image")

    pcd = load_pcd(cloud_xyz[bestcloudidx], labels[bestcloudidx])
    o3d.visualization.draw_geometries([pcd])

    # load intrinsics and extrinsics
    cameraparams = read_yaml(os.path.join(twopath, 'camera_calibration.yaml'))

    # get extrinsics
    camextdata = cameraparams['centre-camera']['extrinsics']

    extT = get_extrinsics_yaml(camextdata)

    # load pose information
    poses2d = pd.read_csv(os.path.join(twopath, 'poses2d.csv'), sep=' ').sort_index()
    poses2d = poses2d.rename(columns={poses2d.columns[0]: "ts"})
    poses2d = poses2d.set_index('ts')

    image_timestamps_datetime = pd.to_datetime(
        [float(ts) for ts in image_timestamp_strings], unit="s"
    )

    this2dpose = poses2d.loc[str(image_timestamps_datetime[idx])]

    q = quaternion.quaternion(this2dpose.qw, this2dpose.qx, this2dpose.qy, this2dpose.qz)
    Rm = quaternion.as_rotation_matrix(q)
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = np.array([this2dpose.x, this2dpose.y, this2dpose.z])

    cloud_wrt_camera = pcd.transform(np.linalg.inv(extT))  # this is the lidar points in the local reference frame.

    # If want to convert points into a global ref frame:
    # test = cloud_wrt_camera.transform(T)

    viewcloud = np.asarray(cloud_wrt_camera.points)

    # pcd_transformed = pcd.transform(global2local)
    # viewcloud = np.asarray(pcd_transformed.points)
    # cloud_wrt_camera = pcd.transform(np.linalg.inv(T))
    # viewcloud = np.asarray(cloud_wrt_camera.points)

    # load raw image and labelimg
    raw_rgb_img = read_rgb_image(images[idx])
    label_img = read_rgb_image(labelimages[idx])

    print('')

    successcount = project_labelcloud_to_image(
        viewcloud.astype(np.float64),
        cameraparams,
        raw_rgb_img,
        label_img,
        image_name=image_timestamp_strings[idx],
        vizoutfolder=None,
    )
    print("Number of lidar points projected: ", successcount)

    plt.show()


def main():
    project_cloud_to_image()
    # ds = WildScenes(seq='K-01')
    # print(len(ds))
    # idx = 0
    # points = ds.get_cloud(idx)
    # img = ds.get_image(idx)
    # print(points.shape)
    # print(np.asarray(img).shape)
    # imgs, rots, trans, intrins, post_rots, post_trans = ds.get_images_data(idx)
    # print(imgs.shape)


if __name__ == "__main__":
    main()
