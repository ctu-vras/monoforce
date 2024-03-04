from __future__ import absolute_import
import os
import numpy as np
from matplotlib import pyplot as plt
from ..utils import position
from ..transformations import transform_cloud
from ..cloudproc import filter_grid
from .data import data_dir
from copy import copy
import torch
import yaml
from PIL import Image
from numpy.lib.recfunctions import unstructured_to_structured
from scipy.spatial.transform import Rotation
import open3d as o3d
from tqdm import tqdm


__all__ = [
    'seq_names',
    'Rellis3D',
]

seq_names = [
    '00000',
    '00001',
    '00002',
    '00003',
    '00004',
]


def read_points_ply(path, dtype=np.float32):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    assert points.shape[1] == 3
    points = unstructured_to_structured(points.astype(dtype=dtype), names=['x', 'y', 'z'])
    del pcd
    return points


def read_points_bin(path, dtype=np.float32):
    xyzi = np.fromfile(path, dtype=dtype)
    xyzi = xyzi.reshape((-1, 4))
    points = unstructured_to_structured(xyzi.astype(dtype=dtype), names=['x', 'y', 'z', 'i'])
    return points


def read_points_labels(path, dtype=np.uint32):
    label = np.fromfile(path, dtype=dtype)
    label = label.reshape((-1, 1))
    # label = convert_label(label, inverse=False)
    label = unstructured_to_structured(label.astype(dtype=dtype), names=['label'])
    return label


def read_points(path, dtype=np.float32):
    # https://stackoverflow.com/questions/5899497/how-can-i-check-the-extension-of-a-file
    if path.lower().endswith('.ply'):
        points = read_points_ply(path, dtype)
    elif path.lower().endswith('.bin'):
        points = read_points_bin(path, dtype)
    else:
        raise ValueError('Cloud file must have .ply or .bin extension')
    return points


def read_poses(path):
    data = np.loadtxt(path)
    poses = np.asarray([np.eye(4) for _ in range(len(data))]).reshape([-1, 4, 4])
    poses[:, :3, :4] = data.reshape([-1, 3, 4])
    return poses


def read_rgb(path):
    img = Image.open(path)
    img = np.asarray(img, dtype=np.uint8)
    return img


def read_intrinsics(path):
    data = np.loadtxt(path)
    K = np.zeros((3, 3))
    K[0, 0] = data[0]
    K[1, 1] = data[1]
    K[2, 2] = 1
    K[0, 2] = data[2]
    K[1, 2] = data[3]
    return K


def read_extrinsics(path, key='os1_cloud_node-pylon_camera_node'):
    """
    Transformation between camera and lidar
    """
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    q = data[key]['q']
    q = np.array([q['x'], q['y'], q['z'], q['w']])
    t = data[key]['t']
    t = np.array([t['x'], t['y'], t['z']])
    R_vc = Rotation.from_quat(q)
    R_vc = R_vc.as_matrix()

    RT = np.eye(4, 4)
    RT[:3, :3] = R_vc
    RT[:3, -1] = t
    return RT


class Rellis3D(torch.utils.data.Dataset):
    def __init__(self, seq=None, path=None):
        """Rellis-3D dataset: https://unmannedlab.github.io/research/RELLIS-3D.

        :param seq: Sequence number (from 0 to 4).
        :param path: Dataset path, takes precedence over name.
        """
        assert isinstance(seq, str) or isinstance(seq, int)
        if isinstance(seq, int):
            seq = '%05d' % seq
        if path is None:
            path = os.path.join(data_dir, 'Rellis3D')

        self.seq = seq
        self.path = path
        self.calib = self.get_calibration()
        self.poses = self.get_poses()
        self.ids_lid, self.ts_lid = self.get_ids(sensor='lidar')
        self.ids_rgb, self.ts_rgb = self.get_ids(sensor='rgb')
        self.ids = self.ids_lid

    def get_calibration(self):
        P = np.zeros([3, 4])
        K = read_intrinsics(self.intrinsics_path())
        P[:3, :3] = K
        calibration = {
            'K': K,
            'P': P,
            'lid2cam': read_extrinsics(self.lidar2cam_path(), key='os1_cloud_node-pylon_camera_node'),
            'robot2lidar': read_extrinsics(self.robot2lidar_path(), key='base_link-os1_cloud_node'),
            'dist_coeff': np.array([-0.134313, -0.025905, 0.002181, 0.00084, 0]),
            'img_width': 1920,
            'img_height': 1200,
        }
        return calibration

    def get_ids(self, sensor='lidar'):
        if sensor == 'lidar':
            sensor_folder = 'os1_cloud_node_color_ply'
        elif sensor == 'rgb':
            sensor_folder = 'pylon_camera_node'
        else:
            raise ValueError('Unsupported sensor type (choose one of: lidar, or rgb, or semseg)')
        # id = frame0000i_sec_msec
        ids = [f[:-4] for f in np.sort(os.listdir(os.path.join(self.path, self.seq, sensor_folder)))]
        ts = [float('%.3f' % (float(id.split('-')[1].split('_')[0]) + float(id.split('-')[1].split('_')[1]) / 1000.0))
              for id in ids]
        ts = np.sort(ts).tolist()
        return ids, ts

    def get_poses(self):
        poses = read_poses(self.cloud_poses_path())
        # transform to robot frame
        Tr = self.calib['robot2lidar']
        # poses = np.einsum("ij,njk->nik", Tr, poses)
        poses = np.asarray([pose @ np.linalg.inv(Tr) for pose in poses])
        return poses

    def lidar_cloud_path(self, id, filetype='bin'):
        if filetype == '.ply':
            return os.path.join(self.path, self.seq, 'os1_cloud_node_color_ply', '%s.ply' % id)
        else:
            return os.path.join(self.path, self.seq, 'os1_cloud_node_kitti_bin', '%06d.bin' % self.ids_lid.index(id))

    def cloud_label_path(self, id):
        return os.path.join(self.path, self.seq, 'os1_cloud_node_semantickitti_label_id',
                            '%06d.label' % self.ids_lid.index(id))

    def cloud_poses_path(self):
        # return os.path.join(self.path, 'calibration', self.seq, self.poses_file)
        return os.path.join(self.path, self.seq, 'poses.txt')

    def image_path(self, id):
        return os.path.join(self.path, self.seq, 'pylon_camera_node', '%s.jpg' % id)

    def semseg_path(self, id):
        return os.path.join(self.path, self.seq, 'pylon_camera_node_label_id', '%s.png' % id)

    def intrinsics_path(self):
        return os.path.join(self.path, 'calibration', self.seq, 'camera_info.txt')

    def lidar2cam_path(self):
        return os.path.join(self.path, 'calibration', self.seq, 'transforms.yaml')

    def robot2lidar_path(self):
        return os.path.join(self.path, 'calibration', 'base_link2os_lidar.yaml')

    def get_sample(self, i):
        id = self.ids[i]
        cloud = self.get_cloud(id)
        pose = self.get_cloud_pose(id)
        img = self.get_image(id)
        K = self.calib['K']
        return cloud, pose, img, K

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get_sample(item)

        ds = copy(self)
        if isinstance(item, (list, tuple)):
            ds.ids = [self.ids[i] for i in item]
        else:
            assert isinstance(item, slice)
            ds.ids = self.ids[item]
        return ds

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_cloud(self, id_lid):
        assert id_lid in self.ids_lid
        cloud = read_points(self.lidar_cloud_path(id_lid))
        # transform to robot frame
        cloud = transform_cloud(cloud, self.calib['robot2lidar'])
        return cloud

    def cloud_label(self, id_lid):
        assert id_lid in self.ids_lid
        return read_points_labels(self.cloud_label_path(id_lid))

    def get_cloud_pose(self, id):
        t = float(id.split('-')[1].split('_')[0]) + float(id.split('-')[1].split('_')[1]) / 1000.0
        i = np.searchsorted(self.ts_lid, t)
        i = np.clip(i, 0, len(self.ids_lid))
        pose = self.poses[i]
        return pose

    def get_image(self, id):
        assert id in self.ids  # these are lidar ids
        t = float(id.split('-')[1].split('_')[0]) + float(id.split('-')[1].split('_')[1]) / 1000.0
        i = np.searchsorted(self.ts_rgb, t)
        i = np.clip(i, 0, len(self.ids_rgb) - 1)
        return read_rgb(self.image_path(self.ids_rgb[i]))


def lidar_map_demo():
    for seq_name in seq_names:
        ds = Rellis3D(seq=seq_name)

        plt.figure()
        plt.title('Trajectory')
        plt.axis('equal')
        plt.plot(ds.poses[:, 0, 3], ds.poses[:, 1, 3], '.')
        plt.grid()
        plt.show()

        clouds = []
        for data in tqdm(ds[::100]):
            cloud, pose, img, K = data
            cloud = filter_grid(cloud, grid_res=0.5)
            cloud = position(cloud)
            cloud = transform_cloud(cloud, pose)

            clouds.append(cloud)
        cloud = np.concatenate(clouds)

        poses_pcd = o3d.geometry.PointCloud()
        poses_pcd.points = o3d.utility.Vector3dVector(ds.poses[:, :3, 3])
        poses_pcd.paint_uniform_color([1, 0, 0])

        cloud_pcd = o3d.geometry.PointCloud()
        cloud_pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])

        o3d.visualization.draw_geometries([cloud_pcd, poses_pcd])


def main():
    lidar_map_demo()


if __name__ == '__main__':
    main()
