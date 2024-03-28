import copy
import os
import matplotlib as mpl
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from numpy.lib.recfunctions import unstructured_to_structured, merge_arrays
from matplotlib import pyplot as plt
from ..models.lss.tools import img_transform, normalize_img
from ..config import DPhysConfig
from ..transformations import transform_cloud
from ..cloudproc import estimate_heightmap, hm_to_cloud
from ..utils import position
from ..cloudproc import filter_grid
from ..imgproc import undistort_image
from ..vis import set_axes_equal
from ..utils import normalize
from .utils import load_calib
import cv2
import albumentations as A
from PIL import Image
from tqdm import tqdm
import open3d as o3d

try:
    mpl.use('TkAgg')
except:
    try:
        mpl.use('QtAgg')
    except:
        print('Cannot set matplotlib backend')
    pass


__all__ = [
    'data_dir',
    'DEMPathData',
    'RobinGas',
    'RobinGasVis',
    'robingas_husky_seq_paths',
    'robingas_marv_seq_paths',
    'robingas_tradr_seq_paths',
    'sim_seq_paths',
    'oru_seq_paths',
]

data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))

robingas_husky_seq_paths = [
    os.path.join(data_dir, 'robingas/data/22-10-27-unhost-final-demo/husky_2022-10-27-15-33-57/'),
    os.path.join(data_dir, 'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-10-33-15/'),
    os.path.join(data_dir, 'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-15-01-44/'),
    os.path.join(data_dir, 'robingas/data/22-09-23-unhost/husky/husky_2022-09-23-12-38-31/'),
    os.path.join(data_dir, 'robingas/data/22-06-30-cimicky_haj/husky_2022-06-30-15-58-37/'),
]
robingas_husky_seq_paths = [os.path.normpath(path) for path in robingas_husky_seq_paths]

robingas_marv_seq_paths = [
    os.path.join(data_dir, 'robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-16-37-03/'),
    os.path.join(data_dir, 'robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34/'),
]
robingas_marv_seq_paths = [os.path.normpath(path) for path in robingas_marv_seq_paths]

robingas_tradr_seq_paths = [
    os.path.join(data_dir, 'robingas/data/22-10-20-unhost/ugv_2022-10-20-14-30-57/'),
    os.path.join(data_dir, 'robingas/data/22-10-20-unhost/ugv_2022-10-20-14-05-42/'),
    os.path.join(data_dir, 'robingas/data/22-10-20-unhost/ugv_2022-10-20-13-58-22/'),
]
robingas_tradr_seq_paths = [os.path.normpath(path) for path in robingas_tradr_seq_paths]

sim_seq_paths = [
    os.path.join(data_dir, 'husky_sim/husky_emptyfarm_2024-01-03-13-36-25'),
    os.path.join(data_dir, 'husky_sim/husky_farmWith1CropRow_2024-01-03-13-52-36'),
    os.path.join(data_dir, 'husky_sim/husky_inspection_2024-01-03-14-06-53'),
    os.path.join(data_dir, 'husky_sim/husky_simcity_2024-01-03-13-55-37'),
    os.path.join(data_dir, 'husky_sim/husky_simcity_dynamic_2024-01-03-13-59-08'),
    os.path.join(data_dir, 'husky_sim/husky_simcity_2024-01-09-17-56-34'),
    os.path.join(data_dir, 'husky_sim/husky_simcity_2024-01-09-17-50-23'),
    os.path.join(data_dir, 'husky_sim/husky_emptyfarm_vegetation_2024-01-09-17-18-46'),
]
sim_seq_paths = [os.path.normpath(path) for path in sim_seq_paths]

oru_seq_paths = [
    os.path.join(data_dir, 'ORU/2024_02_07_Husky_campus_forest_bushes/bags/radarize__2024-02-07-10-47-13_0/'),
]
oru_seq_paths = [os.path.normpath(path) for path in oru_seq_paths]


class DEMPathData(Dataset):
    """
    Class to wrap traversability data generated using lidar odometry.

    The data is stored in the following structure:
    - <path>
        - clouds
            - <id>.npz
            - ...
        - cloud_colors
            - <id>.npz
            - ...
        - images
            - <id>_<camera_name>.png
            - ...
        - trajectories
            - <id>.csv
            - ...
        - calibration
            - cameras
                - <camera_name>.yaml
                - ...
            - transformations.yaml
        - terrain
            - <id>.npy
            - ...
        - poses
            - lidar_poses.csv
            - ...

    A sample of the dataset contains:
    - point cloud (N x 3), where N is the number of points
    - height map (H x W)
    - trajectory (T x 4 x 4), where T is the number of poses
    """

    def __init__(self, path, dphys_cfg=DPhysConfig()):
        super(Dataset, self).__init__()
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))
        self.cloud_path = os.path.join(path, 'clouds')
        # assert os.path.exists(self.cloud_path)
        self.cloud_color_path = os.path.join(path, 'cloud_colors')
        # assert os.path.exists(self.cloud_color_path)
        self.traj_path = os.path.join(path, 'trajectories')
        # global pose of the robot (initial trajectory pose on a map) path (from SLAM)
        self.poses_path = os.path.join(path, 'poses', 'lidar_poses.csv')
        # assert os.path.exists(self.traj_path)
        self.calib_path = os.path.join(path, 'calibration')
        # assert os.path.exists(self.calib_path)
        self.dphys_cfg = dphys_cfg
        self.calib = load_calib(calib_path=self.calib_path)
        self.ids = self.get_ids()
        self.ts, self.poses = self.get_poses(return_stamps=True)

    def get_ids(self):
        ids = [f[:-4] for f in os.listdir(self.cloud_path)]
        ids = np.sort(ids)
        return ids

    @staticmethod
    def pose2mat(pose):
        T = np.eye(4)
        T[:3, :4] = pose.reshape((3, 4))
        return T

    def get_poses(self, return_stamps=False):
        if not os.path.exists(self.poses_path):
            print(f'Poses file {self.poses_path} does not exist')
            return None
        data = np.loadtxt(self.poses_path, delimiter=',', skiprows=1)
        stamps, Ts = data[:, 0], data[:, 1:13]
        lidar_poses = np.asarray([self.pose2mat(pose) for pose in Ts], dtype=np.float32)
        # poses of the robot in the map frame
        Tr_robot_lidar = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr_robot_lidar = np.asarray(Tr_robot_lidar, dtype=np.float32).reshape((4, 4))
        Tr_lidar_robot = np.linalg.inv(Tr_robot_lidar)
        poses = lidar_poses @ Tr_lidar_robot
        if return_stamps:
            return stamps, poses
        return poses

    def get_pose(self, i):
        return self.poses[i]

    def get_traj(self, i):
        ind = self.ids[i]
        Tr_robot_lidar = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr_robot_lidar = np.asarray(Tr_robot_lidar, dtype=np.float32).reshape((4, 4))
        # load data from csv file
        csv_path = os.path.join(self.traj_path, '%s.csv' % ind)
        if os.path.exists(csv_path):
            data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
            stamps, poses = data[:, 0], data[:, 1:13]
            poses = np.asarray([self.pose2mat(pose) for pose in poses])
            # transform to robot frame
            poses = Tr_robot_lidar @ poses
        else:
            # get trajectory as sequence of `n_frames` future poses
            n_frames = 100
            i1 = i + n_frames
            i1 = np.clip(i1, 0, len(self))
            poses = copy.copy(self.poses[i:i1])
            # poses = Tr_robot_lidar @ poses
            poses = np.linalg.inv(poses[0]) @ poses
            # time stamps
            stamps = np.asarray(copy.copy(self.ts[i:i1]))

        traj = {
            'stamps': stamps, 'poses': poses,
        }

        return traj

    def get_states_traj(self, i, start_from_zero=False):
        traj = self.get_traj(i)
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

    def get_raw_cloud(self, i):
        ind = self.ids[i]
        cloud_path = os.path.join(self.cloud_path, '%s.npz' % ind)
        assert os.path.exists(cloud_path), f'Cloud path {cloud_path} does not exist'
        cloud = np.load(cloud_path)['cloud']
        if cloud.ndim == 2:
            cloud = cloud.reshape((-1,))
        return cloud

    def get_cloud(self, i):
        cloud = self.get_raw_cloud(i)
        # remove nans from structured array with fields x, y, z
        cloud = cloud[~np.isnan(cloud['x'])]
        # cloud = cloud[~np.isnan(cloud['y'])]
        # cloud = cloud[~np.isnan(cloud['z'])]

        # move points to robot frame
        Tr = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr = np.asarray(Tr, dtype=float).reshape((4, 4))
        cloud = transform_cloud(cloud, Tr)
        return cloud

    def get_cloud_color(self, i):
        ind = self.ids[i]
        if not os.path.exists(self.cloud_color_path):
            return None
        rgb = np.load(os.path.join(self.cloud_color_path, '%s.npz' % ind))['rgb']
        # convert to structured numpy array with 'r', 'g', 'b' fields
        color = unstructured_to_structured(rgb, names=['r', 'g', 'b'])
        return color

    def get_traj_dphyics_terrain(self, i):
        ind = self.ids[i]
        p = os.path.join(self.path, 'terrain', 'traj', 'dphysics', '%s.npy' % ind)
        terrain = np.load(p)['height']
        return terrain

    def estimated_footprint_traj_points(self, i, robot_size=(0.7, 1.0)):
        # robot footprint points grid
        width, length = robot_size
        x = np.arange(-length / 2, length / 2, self.dphys_cfg.grid_res)
        y = np.arange(-width / 2, width / 2, self.dphys_cfg.grid_res)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        footprint0 = np.stack([x, y, z], axis=-1).reshape((-1, 3))

        Tr_base_link__base_footprint = np.asarray(self.calib['transformations']['T_base_link__base_footprint']['data'],
                                                  dtype=float).reshape((4, 4))
        traj = self.get_traj(i)
        poses = traj['poses']
        poses_footprint = poses @ Tr_base_link__base_footprint

        trajectory_points = []
        for Tr in poses_footprint:
            footprint = transform_cloud(footprint0, Tr)
            trajectory_points.append(footprint)
        trajectory_points = np.concatenate(trajectory_points, axis=0)
        return trajectory_points

    def global_cloud(self, vis=False, cached=True):
        path = os.path.join(self.path, 'global_map.pcd')
        if cached and os.path.exists(path):
            # print('Loading global cloud from file...')
            pcd = o3d.io.read_point_cloud(path)
            global_cloud = np.asarray(pcd.points, dtype=np.float32)
        else:
            # create global cloud
            poses = self.get_poses()
            global_cloud = None
            for i in tqdm(range(len(self))):
                cloud = self.get_cloud(i)
                T = poses[i]
                cloud = transform_cloud(cloud, T)
                points = position(cloud)
                points = filter_grid(points, self.dphys_cfg.grid_res, keep='first', log=False)
                if i == 0:
                    global_cloud = points
                else:
                    global_cloud = np.vstack((global_cloud, points))
            # save global cloud to file
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(global_cloud)
            o3d.io.write_point_cloud(path, pcd)

        if vis:
            # remove nans
            global_cloud_vis = global_cloud[~np.isnan(global_cloud).any(axis=1)]
            # remove height outliers
            heights = global_cloud_vis[:, 2]
            h_min = np.quantile(heights, 0.001)
            h_max = np.quantile(heights, 0.999)
            global_cloud_vis = global_cloud_vis[(global_cloud_vis[:, 2] > h_min) & (global_cloud_vis[:, 2] < h_max)]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(global_cloud_vis)
            o3d.visualization.draw_geometries([pcd])
        return global_cloud

    def global_hm_cloud(self, vis=False):
        poses = self.poses
        # create global heightmap cloud
        global_hm_cloud = []
        for i in tqdm(range(len(self))):
            hm = self.get_lidar_height_map(i)
            hm_cloud = hm_to_cloud(hm[0], self.dphys_cfg, mask=hm[1])
            hm_cloud = transform_cloud(hm_cloud.cpu().numpy(), poses[i])
            global_hm_cloud.append(hm_cloud)
        global_hm_cloud = np.concatenate(global_hm_cloud, axis=0)

        if vis:
            import open3d as o3d
            # plot global cloud with open3d
            hm_pcd = o3d.geometry.PointCloud()
            hm_pcd.points = o3d.utility.Vector3dVector(global_hm_cloud)
            o3d.visualization.draw_geometries([hm_pcd])
        return global_hm_cloud

    def estimate_heightmap(self, points, **kwargs):
        # estimate heightmap from point cloud
        height = estimate_heightmap(points, d_min=self.dphys_cfg.d_min, d_max=self.dphys_cfg.d_max,
                                    grid_res=self.dphys_cfg.grid_res, h_max=self.dphys_cfg.h_max,
                                    hm_interp_method=self.dphys_cfg.hm_interp_method, **kwargs)
        return height

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

    def get_lidar_height_map(self, i, cached=True, dir_name=None, **kwargs):
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
        file_path = os.path.join(dir_name, f'{self.ids[i]}')
        if cached and os.path.exists(file_path):
            lidar_hm = np.load(file_path, allow_pickle=True).item()
        else:
            cloud = self.get_cloud(i)
            points = position(cloud)
            lidar_hm = self.estimate_heightmap(points, **kwargs)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, lidar_hm)
        height = lidar_hm['z']
        mask = lidar_hm['mask']
        heightmap = torch.from_numpy(np.stack([height, mask]))
        return heightmap

    def get_traj_height_map(self, i, method='footprint', cached=True, dir_name=None):
        """
        Get height map from trajectory points.
        :param i: index of the sample
        :param method: method to estimate height map from trajectory points
        :param cached: if True, load height map from file if it exists, otherwise estimate it
        :param dir_name: directory to save/load height map
        :return: height map (2 x H x W), where 2 is the number of channels (z and mask)
        """
        assert method in ['dphysics', 'footprint']
        if dir_name is None:
            dir_name = os.path.join(self.path, 'terrain', 'traj', 'footprint')
        if method == 'dphysics':
            height = self.get_traj_dphyics_terrain(i)
            h, w = int(2 * self.dphys_cfg.d_max // self.dphys_cfg.grid_res), int(2 * self.dphys_cfg.d_max // self.dphys_cfg.grid_res)
            # Optimized height map shape is 256 x 256. We need to crop it to 128 x 128
            H, W = height.shape
            if H == 256 and W == 256:
                # print(f'Height map shape is {H} x {W}). Cropping to 128 x 128')
                # select only the h x w area from the center of the height map
                height = height[int(H // 2 - h // 2):int(H // 2 + h // 2),
                                int(W // 2 - w // 2):int(W // 2 + w // 2)]
            # poses in grid coordinates
            poses = self.get_traj(i)['poses']
            poses_grid = poses[:, :2, 3] / self.dphys_cfg.grid_res + np.asarray([w / 2, h / 2])
            poses_grid = poses_grid.astype(int)
            # crop poses to observation area defined by square grid
            poses_grid = poses_grid[(poses_grid[:, 0] > 0) & (poses_grid[:, 0] < w) &
                                    (poses_grid[:, 1] > 0) & (poses_grid[:, 1] < h)]

            # visited by poses dilated height map area mask
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[poses_grid[:, 0], poses_grid[:, 1]] = 1
            mask = cv2.dilate(mask, kernel, iterations=2)
        else:
            assert method == 'footprint'
            file_path = os.path.join(dir_name, f'{self.ids[i]}.npy')
            if cached and os.path.exists(file_path):
                traj_hm = np.load(file_path, allow_pickle=True).item()
            else:
                traj_points = self.estimated_footprint_traj_points(i)
                traj_hm = self.estimate_heightmap(traj_points, robot_radius=None)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                np.save(file_path, traj_hm)
            height = traj_hm['z']
            mask = traj_hm['mask']
        heightmap = torch.from_numpy(np.stack([height, mask]))
        return heightmap

    def get_sample(self, i):
        cloud = self.get_cloud(i)
        if os.path.exists(self.cloud_color_path):
            color = self.get_cloud_color(i)
            cloud = merge_arrays([cloud, color], flatten=True, usemask=False)
        points = position(cloud)

        traj = self.get_traj(i)
        height = self.estimate_heightmap(points, fill_value=0.)

        return cloud, traj, height

    def __getitem__(self, i):
        if isinstance(i, (int, np.int64)):
            sample = self.get_sample(i)
            return sample

        ds = copy.deepcopy(self)
        if isinstance(i, (list, tuple, np.ndarray)):
            ds.ids = [self.ids[k] for k in i]
            ds.poses = [self.poses[k] for k in i]
        else:
            assert isinstance(i, (slice, range))
            ds.ids = self.ids[i]
            ds.poses = self.poses[i]
        return ds

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ids)


class RobinGas(DEMPathData):
    """
    A dataset for traversability estimation from camera and lidar data.

    A sample of the dataset contains:
    - image (3 x H x W)
    - rotation matrix (3 x 3)
    - translation vector (3)
    - intrinsic matrix (3 x 3)
    - post rotation matrix (3 x 3)
    - post translation vector (3)
    - lidar height map (2 x H x W)
    - trajectory height map (2 x H x W)
    - map pose (4 x 4)
    """

    def __init__(self,
                 path,
                 lss_cfg,
                 dphys_cfg=DPhysConfig(),
                 is_train=False,
                 only_front_hm=False):
        super(RobinGas, self).__init__(path, dphys_cfg)
        self.is_train = is_train
        self.only_front_hm = only_front_hm

        # initialize image augmentations
        self.lss_cfg = lss_cfg
        self.img_augs = self.get_img_augs()

        # get camera names
        self.cameras = self.get_camera_names()[:1] if only_front_hm else self.get_camera_names()

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
                    # A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.5),
                    A.RandomSunFlare(src_radius=100, num_flare_circles_lower=1, num_flare_circles_upper=2, p=0.5),
                    # A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=0.5),
                    A.RandomToneCurve(scale=0.1, p=0.5),
            ])
        else:
            return None

    def get_raw_image(self, i, camera=None):
        if camera is None:
            camera = self.cameras[0]
        ind = self.ids[i]
        img_path = os.path.join(self.path, 'images', '%s_%s.png' % (ind, camera))
        assert os.path.exists(img_path), f'Image path {img_path} does not exist'
        img = Image.open(img_path)
        img = np.asarray(img)
        return img

    def get_raw_img_size(self, i=0, cam=None):
        if cam is None:
            cam = self.cameras[0]
        img = self.get_raw_image(i, cam)
        return img.shape[0], img.shape[1]

    def get_camera_names(self):
        cams_yaml = os.listdir(os.path.join(self.path, 'calibration/cameras'))
        cams = [cam.replace('.yaml', '') for cam in cams_yaml]
        if 'camera_up' in cams:
            cams.remove('camera_up')
        return sorted(cams)

    def get_image(self, i, cam=None, undistort=False):
        if cam is None:
            cam = self.cameras[0]
        img = self.get_raw_image(i, cam)
        for key in self.calib.keys():
            if cam in key:
                cam = key
                break
        K = self.calib[cam]['camera_matrix']['data']
        r, c = self.calib[cam]['camera_matrix']['rows'], self.calib[cam]['camera_matrix']['cols']
        K = np.array(K).reshape((r, c))
        if undistort:
            D = self.calib[cam]['distortion_coefficients']['data']
            D = np.array(D)
            img, K = undistort_image(img, K, D)
        return img, K

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
    
    def get_images_data(self, i, normalize=True):
        imgs = []
        rots = []
        trans = []
        post_rots = []
        post_trans = []
        intrins = []

        for cam in self.cameras:
            img, K = self.get_image(i, cam, undistort=False)
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
            T_robot_cam = self.calib['transformations'][f'T_base_link__{cam}']['data']
            T_robot_cam = np.asarray(T_robot_cam, dtype=np.float32).reshape((4, 4))
            rot = torch.as_tensor(T_robot_cam[:3, :3])
            tran = torch.as_tensor(T_robot_cam[:3, 3])

            imgs.append(img)
            rots.append(rot)
            trans.append(tran)
            intrins.append(K)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        outputs = [torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                   torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans)]
        outputs = [torch.as_tensor(i, dtype=torch.float32) for i in outputs]

        return outputs

    def get_sample(self, i):
        img, rot, tran, intrin, post_rot, post_tran = self.get_images_data(i)
        hm_lidar = self.get_lidar_height_map(i, robot_radius=1.0)
        hm_traj = self.get_traj_height_map(i)
        if self.only_front_hm:
            mask = self.crop_front_height_map(hm_lidar[1:2], only_mask=True)
            hm_lidar[1] = hm_lidar[1] * torch.from_numpy(mask)
            hm_traj[1] = hm_traj[1] * torch.from_numpy(mask)
        map_pose = torch.as_tensor(self.get_pose(i))
        return img, rot, tran, intrin, post_rot, post_tran, hm_lidar, hm_traj, map_pose


class RobinGasVis(RobinGas):
    def __init__(self, path, lss_cfg, dphys_cfg=DPhysConfig(), is_train=True, only_front_hm=False):
        super(RobinGasVis, self).__init__(path, lss_cfg, dphys_cfg=dphys_cfg, is_train=is_train, only_front_hm=only_front_hm)

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_images_data(i)
        hm_lidar = self.get_lidar_height_map(i)
        hm_traj = self.get_traj_height_map(i)
        if self.only_front_hm:
            mask = self.crop_front_height_map(hm_lidar[1:2], only_mask=True)
            hm_lidar[1] = hm_lidar[1] * torch.from_numpy(mask)
            hm_traj[1] = hm_traj[1] * torch.from_numpy(mask)
        map_pose = torch.as_tensor(self.get_pose(i))
        lidar_pts = torch.as_tensor(position(self.get_cloud(i))).T
        return imgs, rots, trans, intrins, post_rots, post_trans, hm_lidar, hm_traj, map_pose, lidar_pts


def heightmap_demo():
    from ..vis import show_cloud_plt
    from ..cloudproc import filter_grid, filter_range
    import matplotlib.pyplot as plt

    # path = robingas_husky_seq_paths[0]
    path = robingas_tradr_seq_paths[0]
    assert os.path.exists(path)

    cfg = DPhysConfig()
    ds = DEMPathData(path, dphys_cfg=cfg)

    i = np.random.choice(range(len(ds)))
    # i = 0
    sample = ds[i]
    cloud, traj, height = sample
    xyz = position(cloud)
    traj = traj['poses'].squeeze()

    xyz = filter_range(xyz, cfg.d_min, cfg.d_max)
    xyz = filter_grid(xyz, cfg.grid_res)

    plt.figure(figsize=(12, 12))
    show_cloud_plt(xyz, markersize=0.4)
    plt.plot(traj[:, 0, 3], traj[:, 1, 3], traj[:, 2, 3], 'ro', markersize=4)
    # visualize height map as a surface
    ax = plt.gca()
    ax.plot_surface(height['x'], height['y'], height['z'],
                    rstride=1, cstride=1, cmap='viridis', edgecolor='none', alpha=0.7)
    plt.show()


def extrinsics_demo():
    from mayavi import mlab
    from ..vis import draw_coord_frames, draw_coord_frame

    # for path in robingas_husky_seq_paths:
    for path in robingas_tradr_seq_paths:
        assert os.path.exists(path)

        cfg = DPhysConfig()
        ds = DEMPathData(path, dphys_cfg=cfg)

        robot_pose = np.eye(4)
        robot_frame = 'base_link'
        lidar_frame = 'os_sensor'

        Tr_robot_lidar = ds.calib['transformations'][f'T_{robot_frame}__{lidar_frame}']['data']
        Tr_robot_lidar = np.asarray(Tr_robot_lidar, dtype=float).reshape((4, 4))

        if 'robingas' in path:
            if 'marv' in path:
                camera_frames = ['camera_left', 'camera_right', 'camera_up', 'camera_fisheye_front',
                                 'camera_fisheye_rear']
            elif 'husky' in path:
                camera_frames = ['camera_left', 'camera_right', 'camera_front', 'camera_rear']
            else:
                camera_frames = ['camera_left', 'camera_right', 'camera_front', 'camera_rear_left', 'camera_rear_right', 'camera_up']
        elif 'sim' in path:
            camera_frames = ['realsense_left', 'realsense_right', 'realsense_front', 'realsense_rear']
        else:
            camera_frames = ['ids_camera']

        cam_poses = []
        for frame in camera_frames:
            T_robot_cam = ds.calib['transformations'][f'T_{robot_frame}__{frame}']['data']
            T_robot_cam = np.asarray(T_robot_cam, dtype=np.float32).reshape((4, 4))

            cam_poses.append(T_robot_cam[np.newaxis])
        cam_poses = np.concatenate(cam_poses, axis=0)

        # draw coordinate frames
        mlab.figure(size=(800, 800))
        draw_coord_frame(robot_pose, scale=0.5)
        draw_coord_frame(Tr_robot_lidar, scale=0.3)
        draw_coord_frames(cam_poses, scale=0.1)
        mlab.show()


def traversed_height_map():
    # path = np.random.choice(robingas_husky_seq_paths)
    path = np.random.choice(robingas_tradr_seq_paths)
    assert os.path.exists(path)

    cfg = DPhysConfig()

    ds = DEMPathData(path, dphys_cfg=cfg)
    i = np.random.choice(range(len(ds)))

    # trajectory poses
    poses = ds.get_traj(i)['poses']
    # point cloud
    cloud = ds.get_cloud(i)
    points = position(cloud)

    img = ds.get_raw_image(i)

    # height map: estimated from point cloud
    heightmap = ds.estimate_heightmap(points)
    height_geom = heightmap['z']
    x_grid, y_grid = heightmap['x'], heightmap['y']
    # height map: optimized from robot-terrain interaction model
    height_rigid = ds.get_traj_dphyics_terrain(i)

    plt.figure(figsize=(12, 12))
    h, w = height_geom.shape
    xy_grid = poses[:, :2, 3] / cfg.grid_res + np.array([w / 2, h / 2])
    plt.subplot(131)
    plt.imshow(height_geom.T, origin='lower', vmin=-0.5, vmax=0.5, cmap='jet')
    plt.plot(xy_grid[:, 0], xy_grid[:, 1], 'rx', markersize=4)
    plt.subplot(132)
    plt.imshow(height_rigid.T, origin='lower', vmin=-0.5, vmax=0.5, cmap='jet')
    plt.plot(xy_grid[:, 0], xy_grid[:, 1], 'rx', markersize=4)
    plt.subplot(133)
    plt.imshow(img)
    plt.show()


def vis_hm_weights():
    cfg = DPhysConfig()

    # circle mask: all points within a circle of radius 1 m are valid
    x_grid = np.arange(0, cfg.d_max, cfg.grid_res)
    y_grid = np.arange(-cfg.d_max / 2., cfg.d_max / 2., cfg.grid_res)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    # distances from the center
    radius = cfg.d_max / 2.
    dist = np.sqrt(x_grid ** 2 + y_grid ** 2)
    # gaussian mask
    weights_est = np.exp(-dist ** 2 / (2. * radius ** 2))
    # weights_est = np.zeros_like(dist)
    # weights_est[dist <= radius] = 1

    # visualize mask in 3D as a surface
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_grid, y_grid, weights_est, cmap='jet')
    set_axes_equal(ax)
    plt.show()


def vis_estimated_height_map():
    cfg = DPhysConfig()
    cfg.grid_res = 0.1
    cfg.d_max = 12.8
    cfg.d_min = 1.
    cfg.h_max = 1.
    # cfg.hm_interp_method = None
    cfg.hm_interp_method = 'nearest'

    # path = np.random.choice(robingas_husky_seq_paths)
    path = np.random.choice(robingas_tradr_seq_paths)
    ds = DEMPathData(path=path, dphys_cfg=cfg)

    i = np.random.choice(range(len(ds)))
    # i = 0
    cloud = ds.get_cloud(i)
    points = position(cloud)

    heightmap, points = ds.estimate_heightmap(points, return_filtered_points=True)
    x_grid, y_grid, height = heightmap['x'], heightmap['y'], heightmap['z']

    # visualize heightmap
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_grid, y_grid, height, cmap='jet', alpha=0.5)
    step = 5
    points_vis = points[::step, :]
    ax.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], s=1, c='k')
    set_axes_equal(ax)
    plt.show()


def global_cloud_demo():
    # paths = robingas_husky_seq_paths
    # paths = robingas_tradr_seq_paths
    paths = oru_seq_paths
    for path in paths:
        ds = DEMPathData(path=path)
        ds.global_cloud(vis=True)


def trajectory_footprint_heightmap():
    import open3d as o3d

    # path = robingas_husky_seq_paths[0]
    path = robingas_tradr_seq_paths[0]
    assert os.path.exists(path)

    cfg = DPhysConfig()
    ds = DEMPathData(path, dphys_cfg=cfg)

    i = np.random.choice(range(len(ds)))
    sample = ds[i]
    cloud, traj, height = sample
    points = position(cloud)

    traj_points = ds.estimated_footprint_traj_points(i)

    lidar_height = ds.estimate_heightmap(points)['z']
    traj_hm = ds.estimate_heightmap(traj_points)
    traj_height = traj_hm['z']
    traj_mask = traj_hm['mask']
    # print('lidar_height', lidar_height.shape)
    # print('traj_height', traj_height.shape)

    plt.figure()
    plt.subplot(131)
    plt.imshow(lidar_height.T, cmap='jet', vmin=-0.5, vmax=0.5, origin='lower')
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(traj_height.T, cmap='jet', vmin=-0.5, vmax=0.5, origin='lower')
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(traj_mask.T, cmap='gray', origin='lower')
    plt.show()

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(traj_points)
    pcd2.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd1, pcd2])


def main():
    heightmap_demo()
    extrinsics_demo()
    traversed_height_map()
    vis_hm_weights()
    vis_estimated_height_map()
    global_cloud_demo()
    trajectory_footprint_heightmap()


if __name__ == '__main__':
    main()
