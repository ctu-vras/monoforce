from __future__ import absolute_import
import os
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from ..models.lss.tools import img_transform, normalize_img
from ..utils import position, read_yaml
from ..transformations import transform_cloud
from ..cloudproc import filter_grid, estimate_heightmap, hm_to_cloud
from ..config import Config
from .data import data_dir, explore_data
from copy import copy
import torch
import yaml
from PIL import Image
from numpy.lib.recfunctions import unstructured_to_structured
from scipy.spatial.transform import Rotation
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import cKDTree


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
    def __init__(self, seq, path=None):
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
            'clearance': 0.254,  # [m], reference: https://clearpathrobotics.com/warthog-unmanned-ground-vehicle-robot/
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

    def get_sample(self, id):
        cloud = self.get_cloud(id)
        pose = self.get_cloud_pose(id)
        img = self.get_image(id)
        K = self.calib['K']
        return cloud, pose, img, K

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if isinstance(item, (int, np.int64)):
            id = self.ids[item]
            return self.get_sample(id)

        ds = copy(self)
        if isinstance(item, (list, tuple)):
            ds.ids = [self.ids[i] for i in item]
        else:
            assert isinstance(item, (slice, range))
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

class Rellis3DTrav(Rellis3D):
    def __init__(self, seq, data_aug_conf, path=None, cfg=Config(), is_train=False):
        super().__init__(seq, path)
        self.cfg = cfg
        self.data_aug_conf = data_aug_conf
        self.is_train = is_train

    def get_traj(self, id, n_frames=100):
        i = self.ids.index(id)
        i0 = np.clip(i, 0, len(self.ids) - n_frames)
        i1 = i0 + n_frames
        poses = copy(self.poses[i0:i1])
        # transform to robot frame
        poses = np.linalg.inv(poses[0]) @ poses
        # take into account robot's clearance
        poses[:, 2, 3] -= self.calib['clearance']
        footprint_poses = poses
        # time stamps
        stamps = copy(self.ts_lid[i0:i1])
        traj = {'poses': footprint_poses, 'stamps': stamps}
        return traj

    def estimated_footprint_traj_points(self, id, robot_size=(1.38, 1.52)):
        traj = self.get_traj(id)
        poses = traj['poses'].copy()

        # robot footprint points grid
        width, length = robot_size
        x = np.arange(-length / 2, length / 2, self.cfg.grid_res)
        y = np.arange(-width / 2, width / 2, self.cfg.grid_res)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        footprint0 = np.stack([x, y, z], axis=-1).reshape((-1, 3))

        Tr_base_link__base_footprint = np.eye(4)
        Tr_base_link__base_footprint[0, 3] = -self.calib['clearance']

        trajectory_footprint = None
        for pose in poses:
            Tr = pose @ Tr_base_link__base_footprint
            footprint = transform_cloud(footprint0, Tr)
            if trajectory_footprint is None:
                trajectory_footprint = footprint
            else:
                tree = cKDTree(trajectory_footprint)
                d, _ = tree.query(footprint)
                footprint = footprint[d > self.cfg.grid_res]
                trajectory_footprint = np.concatenate([trajectory_footprint, footprint], axis=0)
        return trajectory_footprint

    def estimate_heightmap(self, points, **kwargs):
        # estimate heightmap from point cloud
        height = estimate_heightmap(points, d_min=self.cfg.d_min, d_max=self.cfg.d_max,
                                    grid_res=self.cfg.grid_res, h_max=self.cfg.h_max,
                                    hm_interp_method=self.cfg.hm_interp_method, **kwargs)
        return height

    def get_lidar_height_map(self, id, cached=True, **kwargs):
        # height map from point cloud (!!! assumes points are in robot frame)
        interpolation = self.cfg.hm_interp_method if self.cfg.hm_interp_method is not None else 'no_interp'
        dir_path = os.path.join(self.path, self.seq, 'terrain', 'lidar', interpolation)
        file_path = os.path.join(dir_path, '%05d.npy' % self.ids.index(id))
        # if height map was estimated before - load it
        if cached and os.path.exists(file_path):
            # print('Loading height map from file...')
            xyz_mask = np.load(file_path)
        # otherwise - estimate it
        else:
            # print('Estimating and saving height map...')
            cloud = self.get_cloud(id)
            points = position(cloud)
            xyz_mask = self.estimate_heightmap(points, **kwargs)
            # save height map as numpy array
            result = np.zeros((xyz_mask['z'].shape[0], xyz_mask['z'].shape[1]),
                              dtype=[(key, np.float32) for key in xyz_mask.keys()])
            for key in xyz_mask.keys():
                result[key] = xyz_mask[key]
            os.makedirs(dir_path, exist_ok=True)
            np.save(file_path, result)
        heightmap = np.stack([xyz_mask[i] for i in ['z', 'mask']])
        heightmap = np.asarray(heightmap, dtype=np.float32)
        return heightmap

    def get_traj_height_map(self, id, cached=True):
        file_path = os.path.join(self.path, self.seq, 'terrain', 'traj', 'footprint', '%05d.npy' % self.ids.index(id))
        if cached and os.path.exists(file_path):
            traj_hm = np.load(file_path, allow_pickle=True).item()
        else:
            traj_points = self.estimated_footprint_traj_points(id)
            traj_hm = self.estimate_heightmap(traj_points, robot_radius=None)
            # save height map as numpy array
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, traj_hm)
        height = traj_hm['z']
        mask = traj_hm['mask']
        heightmap = np.stack([height, mask])
        return heightmap

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, id, normalize=True):
        img = self.get_image(id)
        K = self.calib['K']
        # if self.is_train:
        #     img = self.img_augs(image=img)['image']

        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        # augmentation (resize, crop, horizontal flip, rotate)
        resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
        img, post_rot2, post_tran2 = img_transform(Image.fromarray(img), post_rot, post_tran,
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
        if normalize:
            img = normalize_img(img)
        else:
            img = torchvision.transforms.ToTensor()(img)
        K = torch.as_tensor(K)

        # extrinsics
        T_robot_lidar = self.calib['robot2lidar']
        T_lidar_cam = self.calib['lid2cam']
        T_robot_cam = T_robot_lidar @ T_lidar_cam
        rot = torch.as_tensor(T_robot_cam[:3, :3])
        tran = torch.as_tensor(T_robot_cam[:3, 3])

        outputs = [img, rot, tran, K, post_rot, post_tran]
        outputs = [torch.as_tensor(i, dtype=torch.float32) for i in outputs]

        return outputs

    def get_sample(self, id):
        inputs = self.get_image_data(id)
        inputs = [i.unsqueeze(0) for i in inputs]
        img, rot, tran, K, post_rot, post_tran = inputs
        hm_lidar = torch.as_tensor(self.get_lidar_height_map(id))
        hm_traj = torch.as_tensor(self.get_traj_height_map(id))
        lidar_pts = torch.as_tensor(position(self.get_cloud(id))).T
        map_pose = torch.as_tensor(self.get_cloud_pose(id))
        return img, rot, tran, K, post_rot, post_tran, hm_lidar, hm_traj, map_pose, lidar_pts


def global_map_demo():
    for seq_name in seq_names:
        ds = Rellis3D(seq=seq_name)

        plt.figure()
        plt.title('Route %s' % seq_name)
        plt.axis('equal')
        plt.plot(ds.poses[:, 0, 3], ds.poses[:, 1, 3], '.')
        plt.grid()
        plt.show()

        clouds = []
        for i in tqdm(range(0, len(ds), 100)):
            cloud, pose, img, K = ds[i]
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


def traversed_cloud_demo():
    cfg = Config()
    config_path = os.path.join(data_dir, '../config/dphys_cfg.yaml')
    assert os.path.isfile(config_path), 'Config file %s does not exist' % config_path
    cfg.from_yaml(config_path)

    lss_cfg_path = os.path.join(data_dir, '../config/lss_cfg.yaml')
    assert os.path.isfile(lss_cfg_path)
    lss_cfg = read_yaml(lss_cfg_path)
    data_aug_conf = lss_cfg['data_aug_conf']

    seq_name = np.random.choice(seq_names)
    ds = Rellis3DTrav(seq=seq_name, cfg=cfg, data_aug_conf=data_aug_conf)

    id = np.random.choice(ds.ids)
    cloud = ds.get_cloud(id)
    points = position(cloud)

    traj = ds.get_traj(id, n_frames=100)
    poses = traj['poses']

    footprint_traj = ds.estimated_footprint_traj_points(id)
    print(cloud.shape, poses.shape, footprint_traj.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd_poses = o3d.geometry.PointCloud()
    pcd_poses.points = o3d.utility.Vector3dVector(poses[:, :3, 3])
    pcd_poses.paint_uniform_color([0, 1, 0])

    pcd_footprint = o3d.geometry.PointCloud()
    pcd_footprint.points = o3d.utility.Vector3dVector(footprint_traj)
    pcd_footprint.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd, pcd_poses, pcd_footprint])


def heightmap_demo():
    cfg = Config()
    p = os.path.join(data_dir, '../config/dphys_cfg.yaml')
    assert os.path.isfile(p), 'Config file %s does not exist' % p
    cfg.from_yaml(p)

    p = os.path.join(data_dir, '../config/lss_cfg.yaml')
    lss_cfg = read_yaml(p)
    data_aug_conf = lss_cfg['data_aug_conf']
    grid_conf = lss_cfg['grid_conf']

    seq_name = np.random.choice(seq_names)
    ds = Rellis3DTrav(seq=seq_name, cfg=cfg, data_aug_conf=data_aug_conf)

    i = np.random.choice(len(ds))
    sample = ds[i]
    for s in sample:
        print(s.shape)

    ds_path = os.path.join(data_dir, 'Rellis3D', seq_name)
    explore_data(ds_path, grid_conf, data_aug_conf, cfg, is_train=False, DataClass=Rellis3DTrav)


def main():
    # global_map_demo()
    # traversed_cloud_demo()
    heightmap_demo()


if __name__ == '__main__':
    main()
