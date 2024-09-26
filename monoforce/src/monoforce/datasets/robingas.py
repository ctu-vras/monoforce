import copy
import os
import matplotlib as mpl
import numpy as np
import torch
import torchvision
from skimage.draw import polygon
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from ..models.terrain_encoder.utils import img_transform, normalize_img, resize_img
from ..models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, sample_augmentation
from ..config import DPhysConfig
from ..transformations import transform_cloud
from ..cloudproc import estimate_heightmap, hm_to_cloud, filter_range
from ..utils import position, timing, read_yaml
from ..cloudproc import filter_grid
from ..imgproc import undistort_image
from ..utils import normalize, load_calib
from .coco import COCO_CATEGORIES
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
    'RobinGasBase',
    'RobinGas',
    'RobinGasPoints',
    'robingas_seq_paths',
]

data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))

robingas_seq_paths = {
    'husky': [
        os.path.join(data_dir, 'RobinGas/husky/husky_2022-10-27-15-33-57'),
        os.path.join(data_dir, 'RobinGas/husky/husky_2022-09-27-10-33-15'),
        os.path.join(data_dir, 'RobinGas/husky/husky_2022-09-27-15-01-44'),
        os.path.join(data_dir, 'RobinGas/husky/husky_2022-09-23-12-38-31'),
        os.path.join(data_dir, 'RobinGas/husky/husky_2022-06-30-15-58-37'),
    ],
    'marv': [
        # os.path.join(data_dir, 'RobinGas/marv/ugv_2022-08-12-16-37-03'),
        # os.path.join(data_dir, 'RobinGas/marv/ugv_2022-08-12-15-18-34'),
        os.path.join(data_dir, 'RobinGas/marv/24-08-14-monoforce-long_drive'),
    ],
    'tradr': [
        os.path.join(data_dir, 'RobinGas/tradr/ugv_2022-10-20-14-30-57'),
        os.path.join(data_dir, 'RobinGas/tradr/ugv_2022-10-20-14-05-42'),
        os.path.join(data_dir, 'RobinGas/tradr/ugv_2022-10-20-13-58-22'),
        # os.path.join(data_dir, 'RobinGas/tradr/ugv_2022-06-30-11-30-57'),
    ],
    'tradr2': [
        os.path.join(data_dir, 'RobinGas/tradr2/ugv_2024-09-10-17-02-31'),
        os.path.join(data_dir, 'RobinGas/tradr2/ugv_2024-09-10-17-12-12'),
    ],
    'husky_oru': [
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2023-08-16-11-02-33_0'),
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2023-08-16-11-09-06_0'),
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2023-08-16-11-24-37_0'),
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2023-08-16-11-37-14_0'),
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2023-08-16-11-44-56_0'),
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2023-08-16-11-54-42_0'),
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2024-02-07-10-47-13_0'),  # no radar
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2024-04-27-15-02-12_0'),
        # os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2024-05-01-15-48-29_0'),  # localization must be fixed
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize__2024-05-24-13-21-28_0'),  # no radar
        os.path.join(data_dir, 'RobinGas/husky_oru/radarize_2024-06-12-10-06-11_0'),  # high grass with radar
    ],
}


class RobinGasBase(Dataset):
    """
    Class to wrap traversability data generated using lidar odometry.

    The data is stored in the following structure:
    - <path>
        - clouds
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
    - trajectory (T x 4 x 4), where the horizon T is the number of poses
    """

    def __init__(self, path, dphys_cfg=DPhysConfig()):
        super(Dataset, self).__init__()
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))
        self.cloud_path = os.path.join(path, 'clouds')
        self.radar_cloud_path = os.path.join(path, 'radar_clouds')
        self.traj_path = os.path.join(path, 'trajectories')
        self.poses_path = os.path.join(path, 'poses', 'lidar_poses.csv')
        self.calib_path = os.path.join(path, 'calibration')
        self.controls_path = os.path.join(path, 'controls', 'tracks_vel.csv')
        self.dphys_cfg = dphys_cfg
        self.calib = load_calib(calib_path=self.calib_path)
        self.ids = self.get_ids()
        self.ts, self.poses = self.get_poses(return_stamps=True)
        self.camera_names = self.get_camera_names()

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
        stamps -= stamps[0]  # start time from 0
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

    def get_controls(self, i):
        if not os.path.exists(self.controls_path):
            print(f'Controls file {self.controls_path} does not exist')
            return None, None

        data = np.loadtxt(self.controls_path, delimiter=',', skiprows=1)
        all_stamps, all_vels = data[:, 0], data[:, 1:]
        all_stamps -= all_stamps[0]  # start time from 0
        time_left = copy.copy(self.ts[i])
        T_horizon, dt = self.dphys_cfg.traj_sim_time, self.dphys_cfg.dt
        time_right = time_left + T_horizon
        # find the closest index to the left and right in all times
        il = np.argmin(np.abs(np.asarray(all_stamps) - time_left))
        ir = np.argmin(np.abs(np.asarray(all_stamps) - time_right))
        ir = max(il + 1, ir)
        ir = np.clip(ir, 0, len(all_vels) - 1)
        timestamps = np.asarray(all_stamps[il:ir])
        timestamps = timestamps - timestamps[0]
        vels = all_vels[il:ir]

        if vels.shape[1] == 4:
            # velocities of the tracks: front left, front right, rear left, rear right
            v_fl, v_fr, v_rl, v_rr = vels[:, 0], vels[:, 1], vels[:, 2], vels[:, 3]

            # average left and right tracks velocities
            v_left = (v_fl + v_rl) / 2
            v_right = (v_fr + v_rr) / 2
            vels = np.stack([v_left, v_right], axis=1)
        else:
            assert vels.shape[1] == 2, f'Velocities array has wrong shape: {vels.shape}'

        # velocities interpolation
        interp_times = np.arange(0.0, time_right - time_left, dt)
        interp_vels = np.zeros((len(interp_times), 2))
        for i in range(vels.shape[1]):
            interp_vels[:, i] = np.interp(interp_times, timestamps, vels[:, i])

        timestamps = interp_times
        vels = interp_vels
        assert len(timestamps) == len(vels), f'Velocity and time stamps have different lengths'
        assert len(timestamps) == int(T_horizon / dt), f'Velocity and time stamps have different lengths'

        return timestamps, np.asarray(vels, dtype=np.float32)

    def get_camera_names(self):
        cams_yaml = os.listdir(os.path.join(self.path, 'calibration/cameras'))
        cams = [cam.replace('.yaml', '') for cam in cams_yaml]
        if 'camera_up' in cams:
            cams.remove('camera_up')
        return sorted(cams)

    def get_traj(self, i, n_frames=10):
        # n_frames equals to the number of future poses (trajectory length)
        T_horizon = self.dphys_cfg.traj_sim_time

        # get trajectory as sequence of `n_frames` future poses
        all_poses = self.get_poses(return_stamps=False)
        all_ts = copy.copy(self.ts)
        il = i
        ir = np.argmin(np.abs(all_ts - (self.ts[i] + T_horizon)))
        ir = min(max(ir, il+1), len(all_ts))
        poses = all_poses[il:ir]
        stamps = np.asarray(all_ts[il:ir])

        # make sure the trajectory has the fixed length
        if len(poses) < n_frames:
            # repeat the last pose to fill the trajectory
            poses = np.concatenate([poses, np.tile(poses[-1:], (n_frames - len(poses), 1, 1))], axis=0)
            dt = np.mean(np.diff(stamps))
            stamps = np.concatenate([stamps, stamps[-1] + np.arange(1, n_frames - len(stamps) + 1) * dt], axis=0)
        # truncate the trajectory
        poses = poses[:n_frames]
        stamps = stamps[:n_frames]
        assert len(poses) == len(stamps), f'Poses and time stamps have different lengths'
        assert len(poses) == n_frames

        # transform poses to the same coordinate frame as the height map
        poses = np.linalg.inv(poses[0]) @ poses
        stamps = stamps - stamps[0]

        traj = {
            'stamps': stamps, 'poses': poses,
        }

        return traj

    def get_states_traj(self, i):
        traj = self.get_traj(i)
        poses = traj['poses']
        tstamps = traj['stamps']

        # transform poses to the same coordinate frame as the height map
        Tr = np.linalg.inv(poses[0])
        poses = np.asarray([np.matmul(Tr, p) for p in poses])
        # count time from 0
        tstamps = tstamps - tstamps[0]

        xs = np.asarray(poses[:, :3, 3])
        Rs = np.asarray(poses[:, :3, :3])

        n_states = len(xs)
        ts = np.asarray(tstamps)

        dps = np.diff(xs, axis=0)
        dt = np.asarray(np.diff(ts), dtype=np.float32).reshape([-1, 1])
        theta = np.arctan2(dps[:, 1], dps[:, 0]).reshape([-1, 1])
        theta = np.concatenate([theta[:1], theta], axis=0)

        xds = np.zeros_like(xs)
        xds[:-1] = dps / dt
        omegas = np.zeros_like(xs)
        omegas[:-1, 2:3] = np.diff(theta, axis=0) / dt  # + torch.diff(angles, dim=0)[:, 2:3] / dt

        states = (xs.reshape([n_states, 3]),
                  xds.reshape([n_states, 3]),
                  Rs.reshape([n_states, 3, 3]),
                  omegas.reshape([n_states, 3]))

        return ts, states

    def get_raw_cloud(self, i):
        ind = self.ids[i]
        cloud_path = os.path.join(self.cloud_path, '%s.npz' % ind)
        assert os.path.exists(cloud_path), f'Cloud path {cloud_path} does not exist'
        cloud = np.load(cloud_path)['cloud']
        if cloud.ndim == 2:
            cloud = cloud.reshape((-1,))
        return cloud

    def get_lidar_cloud(self, i):
        cloud = self.get_raw_cloud(i)
        # remove nans from structured array with fields x, y, z
        cloud = cloud[~np.isnan(cloud['x'])]
        # move points to robot frame
        Tr = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr = np.asarray(Tr, dtype=float).reshape((4, 4))
        cloud = transform_cloud(cloud, Tr)
        return cloud
    
    def get_raw_radar_cloud(self, i):
        ind = self.ids[i]
        cloud_path = os.path.join(self.radar_cloud_path, '%s.npz' % ind)
        assert os.path.exists(cloud_path), f'Cloud path {cloud_path} does not exist'
        cloud = np.load(cloud_path)['cloud']
        if cloud.ndim == 2:
            cloud = cloud.reshape((-1,))
        return cloud
    
    def get_radar_cloud(self, i):
        cloud = self.get_raw_radar_cloud(i)
        # remove nans from structured array with fields x, y, z
        cloud = cloud[~np.isnan(cloud['x'])]
        # close by points contain noise
        cloud = filter_range(cloud, 3.0, np.inf)
        # move points to robot frame
        Tr = self.calib['transformations']['T_base_link__hugin_radar']['data']
        Tr = np.asarray(Tr, dtype=float).reshape((4, 4))
        cloud = transform_cloud(cloud, Tr)
        return cloud

    def get_cloud(self, i, points_source='lidar'):
        assert points_source in ['lidar', 'radar', 'lidar_radar']
        if points_source == 'lidar':
            return self.get_lidar_cloud(i)
        elif points_source == 'radar':
            return self.get_radar_cloud(i)
        else:
            lidar_points = self.get_lidar_cloud(i)
            radar_points = self.get_radar_cloud(i)
            cloud = np.concatenate((lidar_points[['x', 'y', 'z']], radar_points[['x', 'y', 'z']]))
            return cloud

    def get_geom_height_map(self, i, cached=True, dir_name=None, points_source='lidar', **kwargs):
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
        file_path = os.path.join(dir_name, f'{self.ids[i]}.npy')
        if cached and os.path.exists(file_path):
            lidar_hm = np.load(file_path, allow_pickle=True).item()
        else:
            cloud = self.get_cloud(i, points_source=points_source)
            points = position(cloud)
            lidar_hm = self.estimate_heightmap(points, **kwargs)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, lidar_hm)
        height = lidar_hm['z']
        mask = lidar_hm['mask']
        heightmap = torch.from_numpy(np.stack([height, mask]))
        return heightmap

    def get_traj_dphysics_terrain(self, i):
        ind = self.ids[i]
        p = os.path.join(self.path, 'terrain', 'traj', 'dphysics', '%s.npy' % ind)
        terrain = np.load(p)['height']
        return terrain

    def get_footprint_traj_points(self, i, robot_size=(0.7, 1.0)):
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

    def get_global_cloud(self, vis=False, cached=True, save=False, step=1):
        path = os.path.join(self.path, 'map', 'map.pcd')
        if cached and os.path.exists(path):
            # print('Loading global cloud from file...')
            pcd = o3d.io.read_point_cloud(path)
            global_cloud = np.asarray(pcd.points, dtype=np.float32)
        else:
            # create global cloud
            global_cloud = None
            for i in tqdm(range(len(self))[::step]):
                cloud = self.get_cloud(i)
                T = self.get_pose(i)
                cloud = transform_cloud(cloud, T)
                points = position(cloud)
                points = filter_grid(points, self.dphys_cfg.grid_res, keep='first', log=False)
                if i == 0:
                    global_cloud = points
                else:
                    global_cloud = np.vstack((global_cloud, points))
            # save global cloud to file
            if save:
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

            poses = self.get_poses()
            pcd_poses = o3d.geometry.PointCloud()
            pcd_poses.points = o3d.utility.Vector3dVector(poses[:, :3, 3])
            pcd_poses.paint_uniform_color([0.8, 0.1, 0.1])

            # o3d.visualization.draw_geometries([pcd_poses])
            o3d.visualization.draw_geometries([pcd, pcd_poses])
        return global_cloud

    def estimate_heightmap(self, points, **kwargs):
        # estimate heightmap from point cloud
        height = estimate_heightmap(points, d_min=self.dphys_cfg.d_min, d_max=self.dphys_cfg.d_max,
                                    grid_res=self.dphys_cfg.grid_res,
                                    h_max_above_ground=self.dphys_cfg.h_max_above_ground,
                                    robot_clearance=self.calib['clearance'],
                                    hm_interp_method=self.dphys_cfg.hm_interp_method, **kwargs)
        return height

    def get_sample(self, i):
        cloud = self.get_cloud(i)
        traj = self.get_traj(i)
        height = self.estimate_heightmap(position(cloud), fill_value=0.)
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


class RobinGas(RobinGasBase):
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
                 only_front_cam=False,
                 use_rigid_semantics=True):
        super(RobinGas, self).__init__(path, dphys_cfg)
        self.is_train = is_train
        self.only_front_cam = only_front_cam
        self.use_rigid_semantics = use_rigid_semantics
        self.camera_names = self.camera_names[:1] if only_front_cam else self.camera_names

        # initialize image augmentations
        self.lss_cfg = lss_cfg
        self.img_augs = self.get_img_augs()

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
            camera = self.camera_names[0]
        ind = self.ids[i]
        img_path = os.path.join(self.path, 'images', '%s_%s.png' % (ind, camera))
        assert os.path.exists(img_path), f'Image path {img_path} does not exist'
        img = Image.open(img_path)
        return img

    def get_raw_img_size(self, i=0, camera=None):
        if camera is None:
            camera = self.camera_names[0]
        img = self.get_raw_image(i, camera)
        img = np.asarray(img)
        return img.shape[0], img.shape[1]

    def get_image(self, i, camera=None, undistort=False):
        if camera is None:
            camera = self.camera_names[0]
        img = self.get_raw_image(i, camera)
        for key in self.calib.keys():
            if camera in key:
                camera = key
                break
        K = self.calib[camera]['camera_matrix']['data']
        r, c = self.calib[camera]['camera_matrix']['rows'], self.calib[camera]['camera_matrix']['cols']
        K = np.asarray(K, dtype=np.float32).reshape((r, c))
        if undistort:
            D = self.calib[camera]['distortion_coefficients']['data']
            D = np.array(D)
            img = np.asarray(img)
            img, K = undistort_image(img, K, D)
        return img, K

    def get_cached_resized_img(self, i, camera=None):
        cache_dir = os.path.join(self.path, 'images', 'resized')
        os.makedirs(cache_dir, exist_ok=True)
        cached_img_path = os.path.join(cache_dir, '%s_%s.png' % (self.ids[i], camera))
        if os.path.exists(cached_img_path):
            img = Image.open(cached_img_path)
            K = self.calib[camera]['camera_matrix']['data']
            K = np.asarray(K, dtype=np.float32).reshape((3, 3))
            return img, K
        img, K = self.get_image(i, camera)
        img = resize_img(img)
        img.save(cached_img_path)
        return img, K

    def get_images_data(self, i):
        imgs = []
        rots = []
        trans = []
        post_rots = []
        post_trans = []
        intrins = []

        for cam in self.camera_names:
            # img, K = self.get_image(i, cam, undistort=False)
            img, K = self.get_cached_resized_img(i, cam)

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

        img_data = [torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                  torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans)]
        img_data = [torch.as_tensor(i, dtype=torch.float32) for i in img_data]

        return img_data

    def seg_label_to_color(self, seg_label):
        coco_colors = [(np.array(color['color'])).tolist() for color in COCO_CATEGORIES] + [[0, 0, 0]]
        seg_label = np.asarray(seg_label)
        # transform segmentation labels to colors
        size = [s for s in seg_label.shape] + [3]
        seg_color = np.zeros(size, dtype=np.uint8)
        for color_i, color in enumerate(coco_colors):
            seg_color[seg_label == color_i] = color
        return seg_color

    def get_seg_label(self, i, camera=None):
        if camera is None:
            camera = self.camera_names[0]
        id = self.ids[i]
        seg_path = os.path.join(self.path, 'images/seg/', '%s_%s.npy' % (id, camera))
        assert os.path.exists(seg_path), f'Image path {seg_path} does not exist'
        seg = Image.fromarray(np.load(seg_path))
        size = self.get_raw_img_size(i, camera)
        transform = torchvision.transforms.Resize(size)
        seg = transform(seg)
        return seg
    
    def get_semantic_cloud(self, i, classes=None, vis=False, points_source='lidar'):
        coco_classes = [i['name'].replace('-merged', '').replace('-other', '') for i in COCO_CATEGORIES] + ['void']
        if classes is None:
            classes = np.copy(coco_classes)
        # ids of classes in COCO
        selected_labels = []
        for c in classes:
            if c in coco_classes:
                selected_labels.append(coco_classes.index(c))

        lidar_points = position(self.get_cloud(i, points_source=points_source))
        points = []
        labels = []
        for cam in self.camera_names[::-1]:
            seg_label_cam = self.get_seg_label(i, camera=cam)
            seg_label_cam = np.asarray(seg_label_cam)

            K = self.calib[cam]['camera_matrix']['data']
            K = np.asarray(K, dtype=np.float32).reshape((3, 3))
            E = self.calib['transformations'][f'T_base_link__{cam}']['data']
            E = np.asarray(E, dtype=np.float32).reshape((4, 4))
    
            lidar_points = torch.as_tensor(lidar_points)
            E = torch.as_tensor(E)
            K = torch.as_tensor(K)
    
            cam_points = ego_to_cam(lidar_points.T, E[:3, :3], E[:3, 3], K).T
            mask = get_only_in_img_mask(cam_points.T, seg_label_cam.shape[0], seg_label_cam.shape[1])
            cam_points = cam_points[mask]

            # colorize point cloud with values from segmentation image
            uv = cam_points[:, :2].numpy().astype(int)
            seg_label_cam = seg_label_cam[uv[:, 1], uv[:, 0]]
    
            points.append(lidar_points[mask].numpy())
            labels.append(seg_label_cam)

        points = np.concatenate(points)
        labels = np.concatenate(labels)
        colors = self.seg_label_to_color(labels)
        assert len(points) == len(colors)

        # mask out points with labels not in selected classes
        mask = np.isin(labels, selected_labels)
        points = points[mask]
        colors = colors[mask]

        if vis:
            colors = normalize(colors)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd])

        return points, colors

    def global_hm_cloud(self, vis=False):
        # create global heightmap cloud
        global_hm_cloud = []
        for i in tqdm(range(len(self))):
            hm = self.get_geom_height_map(i)
            pose = self.get_pose(i)
            hm_cloud = hm_to_cloud(hm[0], self.dphys_cfg, mask=hm[1])
            hm_cloud = transform_cloud(hm_cloud.cpu().numpy(), pose)
            global_hm_cloud.append(hm_cloud)
        global_hm_cloud = np.concatenate(global_hm_cloud, axis=0)

        if vis:
            # plot global cloud with open3d
            hm_pcd = o3d.geometry.PointCloud()
            hm_pcd.points = o3d.utility.Vector3dVector(global_hm_cloud)
            o3d.visualization.draw_geometries([hm_pcd])
        return global_hm_cloud

    def get_terrain_height_map(self, i, cached=True, dir_name=None, points_source='lidar'):
        """
        Get height map from trajectory points.
        :param i: index of the sample
        :param cached: if True, load height map from file if it exists, otherwise estimate it
        :param dir_name: directory to save/load height map
        :param obstacle_classes: classes of obstacles to include in the height map
        :return: heightmap (2 x H x W), where 2 is the number of channels (z and mask)
        """
        if dir_name is None:
            dir_name = os.path.join(self.path, 'terrain', 'traj', 'footprint')

        file_path = os.path.join(dir_name, f'{self.ids[i]}.npy')
        if cached and os.path.exists(file_path):
            hm_rigid = np.load(file_path, allow_pickle=True).item()
        else:
            traj_points = self.get_footprint_traj_points(i)
            if self.use_rigid_semantics:
                seg_points, _ = self.get_semantic_cloud(i, classes=self.lss_cfg['obstacle_classes'],
                                                        points_source=points_source, vis=False)
                points = np.concatenate((seg_points, traj_points), axis=0)
            else:
                points = traj_points
            hm_rigid = self.estimate_heightmap(points, robot_radius=None)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, hm_rigid)
        height = hm_rigid['z']
        mask = hm_rigid['mask']

        heightmap = torch.from_numpy(np.stack([height, mask]))
        return heightmap

    def front_height_map_mask(self):
        camera = self.camera_names[0]
        K = self.calib[camera]['camera_matrix']['data']
        r, c = self.calib[camera]['camera_matrix']['rows'], self.calib[camera]['camera_matrix']['cols']
        K = np.asarray(K, dtype=np.float32).reshape((r, c))

        # get fov from camera intrinsics
        img_h, img_w = self.lss_cfg['data_aug_conf']['H'], self.lss_cfg['data_aug_conf']['W']
        fx, fy = K[0, 0], K[1, 1]

        fov_x = 2 * np.arctan2(img_h, 2 * fx)
        fov_y = 2 * np.arctan2(img_w, 2 * fy)

        # camera frustum mask
        d = self.dphys_cfg.d_max
        res = self.dphys_cfg.grid_res
        h, w = 2 * d / res, 2 * d / res
        h, w = int(h), int(w)
        mask = np.zeros((h, w), dtype=np.float32)

        to_grid = lambda x: np.array([x[1], x[0]]) / res + np.array([h // 2, w // 2])
        A = to_grid([0, 0])
        B = to_grid([d * np.tan(fov_y / 2), d])
        C = to_grid([-d * np.tan(fov_y / 2), d])

        # select triangle
        rr, cc = polygon([A[0], B[0], C[0]], [A[1], B[1], C[1]], mask.shape)
        mask[rr, cc] = 1.

        return mask

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_images_data(i)
        control_ts, controls = self.get_controls(i)
        traj_ts, states = self.get_states_traj(i)
        Xs, Xds, Rs, Omegas = states
        hm_geom = self.get_geom_height_map(i)
        hm_terrain = self.get_terrain_height_map(i)
        if self.only_front_cam:
            mask = self.front_height_map_mask()
            hm_geom[1] = hm_geom[1] * torch.from_numpy(mask)
            hm_terrain[1] = hm_terrain[1] * torch.from_numpy(mask)
        return (imgs, rots, trans, intrins, post_rots, post_trans,
                hm_geom, hm_terrain,
                control_ts, controls,
                traj_ts, Xs, Xds, Rs, Omegas)


class RobinGasPoints(RobinGas):
    def __init__(self, path, lss_cfg, dphys_cfg=DPhysConfig(), is_train=True,
                 only_front_cam=False, use_rigid_semantics=True, points_source='lidar'):
        super(RobinGasPoints, self).__init__(path, lss_cfg, dphys_cfg=dphys_cfg, is_train=is_train,
                                             only_front_cam=only_front_cam, use_rigid_semantics=use_rigid_semantics)
        assert points_source in ['lidar', 'radar', 'lidar_radar']
        self.points_source = points_source

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_images_data(i)
        control_ts, controls = self.get_controls(i)
        traj_ts, states = self.get_states_traj(i)
        Xs, Xds, Rs, Omegas = states
        hm_geom = self.get_geom_height_map(i, points_source=self.points_source)
        hm_terrain = self.get_terrain_height_map(i, points_source=self.points_source)
        if self.only_front_cam:
            mask = self.front_height_map_mask()
            hm_geom[1] = hm_geom[1] * torch.from_numpy(mask)
            hm_terrain[1] = hm_terrain[1] * torch.from_numpy(mask)
        points = torch.as_tensor(position(self.get_cloud(i, points_source=self.points_source))).T
        return (imgs, rots, trans, intrins, post_rots, post_trans,
                hm_geom, hm_terrain,
                control_ts, controls,
                traj_ts, Xs, Xds, Rs, Omegas,
                points)


def heightmap_demo():
    from ..vis import show_cloud_plt
    from ..cloudproc import filter_grid, filter_range
    import matplotlib.pyplot as plt

    robot = 'husky_oru'
    path = robingas_seq_paths[robot][0]
    assert os.path.exists(path)

    cfg = DPhysConfig()
    ds = RobinGasBase(path, dphys_cfg=cfg)

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
    ax.plot_surface(height['y'], height['x'], height['z'],
                    rstride=1, cstride=1, cmap='viridis', edgecolor='none', alpha=0.7)
    plt.show()


def extrinsics_demo():
    from mayavi import mlab
    from ..vis import draw_coord_frames, draw_coord_frame

    robot = 'husky_oru'
    for path in robingas_seq_paths[robot]:
        assert os.path.exists(path)

        cfg = DPhysConfig()
        ds = RobinGasBase(path, dphys_cfg=cfg)

        robot_pose = np.eye(4)
        robot_frame = 'base_link'
        lidar_frame = 'os_sensor'

        Tr_robot_lidar = ds.calib['transformations'][f'T_{robot_frame}__{lidar_frame}']['data']
        Tr_robot_lidar = np.asarray(Tr_robot_lidar, dtype=float).reshape((4, 4))

        cam_poses = []
        for frame in ds.camera_names:
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
    robot = 'husky_oru'
    path = np.random.choice(robingas_seq_paths[robot])
    assert os.path.exists(path)

    dphys_cfg = DPhysConfig()
    dphys_cfg.from_yaml(os.path.join(data_dir, '../config/dphys_cfg.yaml'))

    lss_cfg = read_yaml(os.path.join(data_dir, f'../config/lss_cfg_{robot}.yaml'))

    ds = RobinGas(path, dphys_cfg=dphys_cfg, lss_cfg=lss_cfg)
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
    height_terrain = ds.get_terrain_height_map(i)[0]

    plt.figure(figsize=(12, 12))
    h, w = height_geom.shape
    xy_grid = poses[:, :2, 3] / dphys_cfg.grid_res + np.array([w / 2, h / 2])
    plt.subplot(131)
    plt.imshow(height_geom.T, origin='lower', vmin=-1, vmax=1, cmap='jet')
    plt.plot(xy_grid[:, 0], xy_grid[:, 1], 'rx', markersize=4)
    plt.subplot(132)
    plt.imshow(height_terrain.T, origin='lower', vmin=-1, vmax=1, cmap='jet')
    plt.plot(xy_grid[:, 0], xy_grid[:, 1], 'rx', markersize=4)
    plt.subplot(133)
    plt.imshow(img)
    plt.show()


def vis_hm_weights():
    from ..vis import set_axes_equal

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
    from ..vis import set_axes_equal

    cfg = DPhysConfig()
    cfg.grid_res = 0.1
    cfg.d_max = 12.8
    cfg.d_min = 1.
    cfg.h_max_above_ground = 1.
    # cfg.hm_interp_method = None
    cfg.hm_interp_method = 'nearest'

    # path = np.random.choice(robingas_seq_paths['husky'])
    path = np.random.choice(robingas_seq_paths['tradr'])
    ds = RobinGasBase(path=path, dphys_cfg=cfg)

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
    robot = 'husky_oru'
    assert robot in ['husky_oru', 'tradr', 'husky']
    paths = robingas_seq_paths[robot]
    for path in paths:
        ds = RobinGasBase(path=path)
        ds.get_global_cloud(vis=True, cached=False, step=10)


def trajectory_footprint_heightmap():
    robot = 'husky_oru'
    for path in robingas_seq_paths[robot]:
        assert os.path.exists(path)

        cfg = DPhysConfig()
        ds = RobinGasBase(path, dphys_cfg=cfg)

        i = np.random.choice(range(len(ds)))
        sample = ds[i]
        cloud, traj, height = sample
        points = position(cloud)

        traj_points = ds.get_footprint_traj_points(i)
        lidar_height = ds.estimate_heightmap(points)['z']

        traj_hm = ds.estimate_heightmap(traj_points)
        traj_height = traj_hm['z']
        traj_mask = traj_hm['mask']

        # plt.figure(figsize=(12, 6))
        # plt.subplot(131)
        # plt.imshow(lidar_height.T, cmap='jet', vmin=-1., vmax=1., origin='lower')
        # plt.colorbar()
        # plt.subplot(132)
        # plt.imshow(traj_height.T, cmap='jet', vmin=-1., vmax=1., origin='lower')
        # plt.colorbar()
        # plt.subplot(133)
        # plt.imshow(traj_mask.T, cmap='gray', origin='lower')
        # plt.show()

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
