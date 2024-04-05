from __future__ import absolute_import
import os
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from ..models.lss.tools import img_transform, normalize_img
from ..utils import position, read_yaml
from ..transformations import transform_cloud
from ..cloudproc import filter_grid, estimate_heightmap, hm_to_cloud
from ..config import DPhysConfig
from .robingas import data_dir
from monoforce.datasets.utils import explore_data
from copy import copy
import torch
import yaml
from PIL import Image
from numpy.lib.recfunctions import unstructured_to_structured
from scipy.spatial.transform import Rotation
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import cKDTree
import albumentations as A


__all__ = [
    'Rellis3DBase',
    'Rellis3D',
    'Rellis3DVis',
    'rellis3d_seq_paths',
]

seq_names = [
    '00000',
    '00001',
    '00002',
    '00003',
    '00004',
]

rellis3d_seq_paths = [
    os.path.join(data_dir, 'Rellis3D', seq_name) for seq_name in seq_names
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


class Rellis3DBase(torch.utils.data.Dataset):
    def __init__(self, path):
        """Rellis-3D dataset: https://unmannedlab.github.io/research/RELLIS-3D.

        :param path: Dataset path
        """
        self.path = path
        self.seq = os.path.basename(path)
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
        ids = [f[:-4] for f in np.sort(os.listdir(os.path.join(self.path, sensor_folder)))]
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
            return os.path.join(self.path, 'os1_cloud_node_color_ply', '%s.ply' % id)
        else:
            return os.path.join(self.path, 'os1_cloud_node_kitti_bin', '%06d.bin' % self.ids_lid.index(id))

    def cloud_label_path(self, id):
        return os.path.join(self.path, 'os1_cloud_node_semantickitti_label_id',
                            '%06d.label' % self.ids_lid.index(id))

    def cloud_poses_path(self):
        return os.path.join(self.path, 'poses.txt')

    def image_path(self, id):
        return os.path.join(self.path, 'pylon_camera_node', '%s.jpg' % id)

    def semseg_path(self, id):
        return os.path.join(self.path, 'pylon_camera_node_label_id', '%s.png' % id)

    def intrinsics_path(self):
        return os.path.join(self.path, '../calibration', self.seq, 'camera_info.txt')

    def lidar2cam_path(self):
        return os.path.join(self.path, '../calibration', self.seq, 'transforms.yaml')

    def robot2lidar_path(self):
        return os.path.join(self.path, '../calibration', 'base_link2os_lidar.yaml')

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
        if isinstance(item, (list, tuple, np.ndarray)):
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

    def get_raw_img_size(self, id=None):
        if id is None:
            id = self.ids[0]
        assert id in self.ids
        img = self.get_image(id)
        return img.shape[0], img.shape[1]


class Rellis3D(Rellis3DBase):
    def __init__(self, path, lss_cfg, dphys_cfg=DPhysConfig(), is_train=False, only_front_hm=False):
        super().__init__(path)
        self.dphys_cfg = dphys_cfg
        self.is_train = is_train
        self.lss_cfg = lss_cfg
        self.img_augs = self.get_img_augs()
        self.only_front_hm = only_front_hm
        self.cameras = ['camera_front']

    def get_img_augs(self):
        if self.is_train:
            return A.Compose([
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, always_apply=False, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    A.Blur(blur_limit=7, p=0.5),
                    A.GaussNoise(var_limit=(10, 50), p=0.5),
                    A.MotionBlur(blur_limit=7, p=0.5),
                    A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200),
                                 p=0.5),
                    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.5),
                    A.RandomSunFlare(src_radius=100, num_flare_circles_lower=1, num_flare_circles_upper=2, p=0.5),
                    A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=0.5),
                    A.RandomToneCurve(scale=0.1, p=0.5),
            ])
        else:
            return None

    def get_traj(self, id, n_frames=100):
        i0 = self.ids.index(id)
        i1 = i0 + n_frames
        i1 = np.clip(i1, 0, len(self.ids))
        poses = copy(self.poses[i0:i1])
        # transform to robot frame
        poses = np.linalg.inv(poses[0]) @ poses
        # take into account robot's clearance
        poses[:, 2, 3] -= self.calib['clearance']
        footprint_poses = poses
        # time stamps
        stamps = np.asarray(copy(self.ts_lid[i0:i1]))
        traj = {'poses': footprint_poses, 'stamps': stamps}
        return traj

    def get_states_traj(self, id, start_from_zero=False):
        traj = self.get_traj(id)
        poses = traj['poses']

        if start_from_zero:
            # transform poses to the same coordinate frame as the height map
            Tr = np.linalg.inv(poses[0])
            poses = np.asarray([np.matmul(Tr, p) for p in poses])
            poses[:, 2, 3] -= self.calib['clearance']
            # count time from 0
            tstamps = traj['stamps']
            tstamps = tstamps - tstamps[0]

        poses = np.asarray(poses, dtype=np.float32)
        tstamps = np.asarray(tstamps, dtype=np.float32)

        xyz = torch.as_tensor(poses[:, :3, 3])
        rot = torch.as_tensor(poses[:, :3, :3])

        n_states = len(xyz)
        tt = torch.tensor(tstamps)[None].T

        dps = torch.diff(xyz, dim=0)
        dt = torch.diff(tt, dim=0)
        theta = torch.atan2(dps[:, 1], dps[:, 0]).view(-1, 1)
        theta = torch.cat([theta[:1], theta], dim=0)

        vel = torch.zeros_like(xyz)
        vel[:-1] = dps / dt
        omega = torch.zeros_like(xyz)
        omega[:-1, 2:3] = torch.diff(theta, dim=0) / dt  # + torch.diff(angles, dim=0)[:, 2:3] / dt

        forces = torch.zeros((n_states, 3, 10))
        states = (xyz.view(n_states, 3, 1),
                  rot.view(n_states, 3, 3),
                  vel.view(n_states, 3, 1),
                  omega.view(n_states, 3, 1),
                  forces.view(n_states, 3, 10))
        return states

    def estimated_footprint_traj_points(self, id, robot_size=(1.38, 1.52)):
        traj = self.get_traj(id)
        poses = traj['poses'].copy()

        # robot footprint points grid
        width, length = robot_size
        x = np.arange(-length / 2, length / 2, self.dphys_cfg.grid_res)
        y = np.arange(-width / 2, width / 2, self.dphys_cfg.grid_res)
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
                footprint = footprint[d > self.dphys_cfg.grid_res]
                trajectory_footprint = np.concatenate([trajectory_footprint, footprint], axis=0)
        return trajectory_footprint

    def estimate_heightmap(self, points, **kwargs):
        # estimate heightmap from point cloud
        height = estimate_heightmap(points, d_min=self.dphys_cfg.d_min, d_max=self.dphys_cfg.d_max,
                                    grid_res=self.dphys_cfg.grid_res, h_max=self.dphys_cfg.h_max,
                                    hm_interp_method=self.dphys_cfg.hm_interp_method, **kwargs)
        return height

    def get_lidar_height_map(self, id, cached=True, dir_name=None, **kwargs):
        """
        Get height map from lidar point cloud.
        :param i: index of the sample
        :param cached: if True, load height map from file if it exists, otherwise estimate it
        :param dir_name: directory to save/load height map
        :param kwargs: additional arguments for height map estimation
        :return: height map (2 x H x W), where 2 is the number of channels (z and mask)
        """
        if dir_name is None:
            dir_name = os.path.join(self.path, 'terrain', 'lidar')
        file_path = os.path.join(dir_name, '%05d.npy' % self.ids.index(id))
        if cached and os.path.exists(file_path):
            lidar_hm = np.load(file_path, allow_pickle=True).item()
        else:
            cloud = self.get_cloud(id)
            points = position(cloud)
            lidar_hm = self.estimate_heightmap(points, **kwargs)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, lidar_hm)
        height = lidar_hm['z']
        # masking out the front part of the height map
        mask = lidar_hm['mask'] * self.crop_front_height_map(height[None], only_mask=True)
        heightmap = torch.from_numpy(np.stack([height, mask]))
        return heightmap

    def get_traj_height_map(self, id, cached=True, dir_name=None):
        """
        Get height map from trajectory points.
        :param i: index of the sample
        :param method: method to estimate height map from trajectory points
        :param cached: if True, load height map from file if it exists, otherwise estimate it
        :param dir_name: directory to save/load height map
        :return: height map (2 x H x W), where 2 is the number of channels (z and mask)
        """
        if dir_name is None:
            dir_name = os.path.join(self.path, 'terrain', 'traj', 'footprint')
        file_path = os.path.join(dir_name, '%05d.npy' % self.ids.index(id))
        if cached and os.path.exists(file_path):
            traj_hm = np.load(file_path, allow_pickle=True).item()
        else:
            traj_points = self.estimated_footprint_traj_points(id)
            traj_hm = self.estimate_heightmap(traj_points, robot_radius=None)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, traj_hm)
        height = traj_hm['z']
        # masking out the front part of the height map
        mask = traj_hm['mask'] * self.crop_front_height_map(height[None], only_mask=True)
        heightmap = torch.from_numpy(np.stack([height, mask]))
        return heightmap

    def crop_front_height_map(self, hm, only_mask=False):
        # square defining observation area on the ground
        square = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]]) * self.dphys_cfg.d_max / 2
        offset = np.asarray([0, self.dphys_cfg.d_max / 2, 0])
        square = square + offset
        h, w = hm.shape[1], hm.shape[2]
        square_grid = square[:, :2] / self.dphys_cfg.grid_res + np.asarray([w / 2, h / 2])
        if only_mask:
            mask = np.zeros((h, w), dtype=np.float32)
            mask[int(square_grid[0, 1]):int(square_grid[2, 1]),
                 int(square_grid[0, 0]):int(square_grid[2, 0])] = 1.
            return mask
        # crop height map to observation area defined by square grid
        hm_front = hm[:, int(square_grid[0, 1]):int(square_grid[2, 1]),
                         int(square_grid[0, 0]):int(square_grid[2, 0])]
        return hm_front

    def sample_augmentation(self):
        H, W = self.get_raw_img_size()
        fH, fW = self.lss_cfg['data_aug_conf']['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.lss_cfg['data_aug_conf']['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.lss_cfg['data_aug_conf']['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.lss_cfg['data_aug_conf']['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.lss_cfg['data_aug_conf']['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.lss_cfg['data_aug_conf']['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, id, normalize=True):
        img = self.get_image(id)
        K = self.calib['K']

        if self.is_train:
            img = self.img_augs(image=img)['image']

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
        if self.only_front_hm:
            hm_lidar = self.crop_front_height_map(hm_lidar)
            hm_traj = self.crop_front_height_map(hm_traj)
        map_pose = torch.as_tensor(self.get_cloud_pose(id))
        return img, rot, tran, K, post_rot, post_tran, hm_lidar, hm_traj, map_pose


class Rellis3DVis(Rellis3D):
    def __init__(self, path, lss_cfg, dphys_cfg=DPhysConfig(), is_train=False, only_front_hm=False):
        super().__init__(path, lss_cfg, dphys_cfg, is_train, only_front_hm)

    def get_sample(self, id):
        inputs = self.get_image_data(id, normalize=False)
        inputs = [i.unsqueeze(0) for i in inputs]
        img, rot, tran, K, post_rot, post_tran = inputs
        hm_lidar = torch.as_tensor(self.get_lidar_height_map(id))
        hm_traj = torch.as_tensor(self.get_traj_height_map(id))
        if self.only_front_hm:
            hm_lidar = self.crop_front_height_map(hm_lidar)
            hm_traj = self.crop_front_height_map(hm_traj)
        lidar_pts = torch.as_tensor(position(self.get_cloud(id))).T
        map_pose = torch.as_tensor(self.get_cloud_pose(id))
        return img, rot, tran, K, post_rot, post_tran, hm_lidar, hm_traj, map_pose, lidar_pts


def global_map_demo():
    for path in rellis3d_seq_paths:
        ds = Rellis3DBase(path=path)

        plt.figure()
        plt.title('Route %s' % path)
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
    dphys_cfg = DPhysConfig()
    config_path = os.path.join(data_dir, '../config/dphys_cfg.yaml')
    assert os.path.isfile(config_path), 'Config file %s does not exist' % config_path
    dphys_cfg.from_yaml(config_path)

    lss_cfg_path = os.path.join(data_dir, '../config/lss_cfg_tradr.yaml')
    assert os.path.isfile(lss_cfg_path)
    lss_cfg = read_yaml(lss_cfg_path)

    path = np.random.choice(rellis3d_seq_paths)
    ds = Rellis3D(path=path, dphys_cfg=dphys_cfg, lss_cfg=lss_cfg)

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


def main():
    global_map_demo()
    traversed_cloud_demo()


if __name__ == '__main__':
    main()
