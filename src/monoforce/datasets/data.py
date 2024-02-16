import os
import matplotlib as mpl
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from numpy.lib.recfunctions import unstructured_to_structured, merge_arrays
from matplotlib import cm, pyplot as plt
from mayavi import mlab
from ..models.lss.tools import ego_to_cam, get_only_in_img_mask, denormalize_img, img_transform, normalize_img
from ..models.lss.model import compile_model
from ..config import Config
from ..transformations import transform_cloud
from ..cloudproc import estimate_heightmap, hm_to_cloud
from ..utils import position, color
from ..cloudproc import filter_grid, filter_range
from ..imgproc import undistort_image, project_cloud_to_image
from ..vis import show_cloud, draw_coord_frame, draw_coord_frames, set_axes_equal
from ..utils import normalize
from .utils import load_cam_calib
import cv2
import albumentations as A
from PIL import Image
from tqdm import tqdm

__all__ = [
    'DEMPathData',
    'RigidDEMPathData',
    'MonoDEMData',
    'OmniDEMData',
    'OmniDEMDataVis',
    'OmniRigidDEMData',
    'OmniRigidDEMDataVis',
    'DepthDEMData',
    'DepthDEMDataVis',
    'TravData',
    'TravDataVis',
    'seq_paths',
    'sim_seq_paths',
]

IGNORE_LABEL = 255
data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))

seq_paths = [
        os.path.join(data_dir, 'robingas/data/22-10-27-unhost-final-demo/husky_2022-10-27-15-33-57_trav/'),
        os.path.join(data_dir, 'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-10-33-15_trav/'),
        os.path.join(data_dir, 'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-15-01-44_trav/'),
        os.path.join(data_dir, 'robingas/data/22-09-23-unhost/husky/husky_2022-09-23-12-38-31_trav/'),
        os.path.join(data_dir, 'robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-16-37-03_trav/'),
        os.path.join(data_dir, 'robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'),
]
seq_paths = [os.path.normpath(path) for path in seq_paths]

sim_seq_paths = [
        os.path.join(data_dir, 'husky_sim/husky_emptyfarm_2024-01-03-13-36-25_trav'),
        os.path.join(data_dir, 'husky_sim/husky_farmWith1CropRow_2024-01-03-13-52-36_trav'),
        os.path.join(data_dir, 'husky_sim/husky_inspection_2024-01-03-14-06-53_trav'),
        os.path.join(data_dir, 'husky_sim/husky_simcity_2024-01-03-13-55-37_trav'),
        os.path.join(data_dir, 'husky_sim/husky_simcity_dynamic_2024-01-03-13-59-08_trav'),
        os.path.join(data_dir, 'husky_sim/husky_simcity_2024-01-09-17-56-34_trav'),
        os.path.join(data_dir, 'husky_sim/husky_simcity_2024-01-09-17-50-23_trav'),
        os.path.join(data_dir, 'husky_sim/husky_emptyfarm_vegetation_2024-01-09-17-18-46_trav'),
]
sim_seq_paths = [os.path.normpath(path) for path in sim_seq_paths]

class DEMPathData(Dataset):
    """
    Class to wrap semi-supervised traversability data generated using lidar odometry.
    Please, have a look at the `save_clouds_and_trajectories_from_bag` script for data generation from bag file.
    The dataset additionally contains camera images, calibration data, IMU measurements,
    and RGB colors projected from cameras onto the point clouds.

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
        - traj_poses.csv

    A sample of the dataset contains:
    - point cloud (N x 3), TODO: describe fields
    - height map (H x W)
    - trajectory (T x 4 x 4)
    """

    def __init__(self, path, cfg=Config()):
        super(Dataset, self).__init__()
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))
        self.cloud_path = os.path.join(path, 'clouds')
        # assert os.path.exists(self.cloud_path)
        self.cloud_color_path = os.path.join(path, 'cloud_colors')
        # assert os.path.exists(self.cloud_color_path)
        self.traj_path = os.path.join(path, 'trajectories')
        # global pose of the robot (initial trajectory pose on a map) path (from SLAM)
        self.poses_path = os.path.join(path, 'traj_poses.csv')
        # assert os.path.exists(self.traj_path)
        self.calib_path = os.path.join(path, 'calibration')
        # assert os.path.exists(self.calib_path)
        self.cfg = cfg
        self.calib = load_cam_calib(calib_path=self.calib_path)
        self.ids = self.get_ids()
        self.poses = self.get_poses()
        self.hm_interp_method = self.cfg.hm_interp_method

    def get_ids(self):
        ids = [f[:-4] for f in os.listdir(self.cloud_path)]
        ids = np.sort(ids)
        return ids

    @staticmethod
    def pose2mat(pose):
        T = np.eye(4)
        T[:3, :4] = pose.reshape((3, 4))
        return T

    def get_poses(self):
        if not os.path.exists(self.poses_path):
            print(f'Trajectory poses file {self.poses_path} does not exist')
            return None
        data = np.loadtxt(self.poses_path, delimiter=',', skiprows=1)
        stamps, Ts = data[:, 0], data[:, 1:13]
        poses = np.asarray([self.pose2mat(pose) for pose in Ts], dtype=np.float32)
        # poses of the robot in the map frame
        Tr = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr = np.asarray(Tr, dtype=np.float32).reshape((4, 4))
        poses = np.asarray([pose @ np.linalg.inv(Tr) for pose in poses])
        return poses

    def get_pose(self, i):
        return self.poses[i]

    def get_traj(self, i):
        ind = self.ids[i]
        # load data from csv file
        csv_path = os.path.join(self.traj_path, '%s.csv' % ind)
        if os.path.exists(csv_path):
            data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
            stamps, poses = data[:, 0], data[:, 1:13]
            # TODO: vels and omegas from data set are not ready yet to be used
            # vels, omegas = data[:, 13:16], data[:, 16:19]
            # accs, betas = data[:, 19:22], data[:, 22:25]
            poses = np.asarray([self.pose2mat(pose) for pose in poses])
        else:
            poses = np.load(os.path.join(self.traj_path, '%s.npz' % ind))['traj']
            dt = 0.5
            stamps = dt * np.arange(len(poses))

        traj = {
            'stamps': stamps, 'poses': poses,
            # 'vels': np.zeros_like(poses), 'omegas': np.zeros_like(poses),
            # 'accs': np.zeros_like(poses), 'betas': np.zeros_like(poses)
        }

        # transform to robot frame
        Tr = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr = np.asarray(Tr, dtype=np.float32).reshape((4, 4))
        traj['poses'] = np.asarray([Tr @ pose for pose in traj['poses']])

        return traj

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

    def get_raw_image(self, i, camera='front'):
        ind = self.ids[i]
        img_path = os.path.join(self.path, 'images', '%s_%s.png' % (ind, camera))
        assert os.path.exists(img_path), f'Image path {img_path} does not exist'
        img = Image.open(img_path)
        img = np.asarray(img)
        return img

    def get_optimized_terrain(self, i):
        ind = self.ids[i]
        p = os.path.join(self.path, 'terrain', '%s.npy' % ind)
        terrain = np.load(p)['height']
        return terrain

    def estimated_footprint_traj_points(self, i, robot_size=(0.7, 1.0)):
        traj = self.get_traj(i)
        poses = traj['poses'].squeeze()

        # robot footprint points grid
        width, length = robot_size
        x = np.arange(-length / 2, length / 2, self.cfg.grid_res)
        y = np.arange(-width / 2, width / 2, self.cfg.grid_res)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        footprint0 = np.stack([x, y, z], axis=-1).reshape((-1, 3))

        Tr_base_link__base_footprint = np.asarray(self.calib['transformations']['T_base_link__base_footprint']['data'],
                                                  dtype=float).reshape((4, 4))
        trajectory_footprint = []
        for pose in poses:
            Tr = pose @ Tr_base_link__base_footprint
            footprint = transform_cloud(footprint0, Tr)
            trajectory_footprint.append(footprint)
        trajectory_footprint = np.concatenate(trajectory_footprint, axis=0)
        return trajectory_footprint

    def get_footprint_terrain(self, i, robot_size=(0.7, 1.0)):
        traj_points = self.estimated_footprint_traj_points(i, robot_size=robot_size)
        traj_hm = self.estimate_heightmap(traj_points)['z']
        return traj_hm

    def global_cloud(self, colorize=False, vis=False):
        poses = self.get_poses()

        # create global cloud
        for i in tqdm(range(len(self))):
            cloud = self.get_cloud(i)
            if colorize:
                # cloud color
                color_struct = self.get_cloud_color(i)
                rgb = normalize(color(color_struct))
            T = poses[i]
            cloud = transform_cloud(cloud, T)
            points = position(cloud)
            if i == 0:
                mask = filter_grid(points, self.cfg.grid_res, keep='first', log=False, only_mask=True)
                global_cloud = points[mask]
                global_cloud_rgb = rgb[mask] if colorize else None
            else:
                global_cloud = np.vstack((global_cloud, points[mask]))
                global_cloud_rgb = np.vstack((global_cloud_rgb, rgb[mask])) if colorize else None

        if vis:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(global_cloud)
            if colorize:
                pcd.colors = o3d.utility.Vector3dVector(global_cloud_rgb)
            o3d.visualization.draw_geometries([pcd])
        return global_cloud

    def estimate_heightmap(self, points, **kwargs):
        # estimate heightmap from point cloud
        height = estimate_heightmap(points, d_min=self.cfg.d_min, d_max=self.cfg.d_max,
                                    grid_res=self.cfg.grid_res, h_max=self.cfg.h_max,
                                    hm_interp_method=self.hm_interp_method, **kwargs)
        return height

    def __getitem__(self, i, visualize=False):
        cloud = self.get_cloud(i)
        if os.path.exists(self.cloud_color_path):
            color = self.get_cloud_color(i)
            cloud = merge_arrays([cloud, color], flatten=True, usemask=False)
        points = position(cloud)

        if visualize:
            trav = np.asarray(cloud['traversability'], dtype=points.dtype)
            valid = trav != IGNORE_LABEL
            show_cloud(points, min=trav[valid].min(), max=trav[valid].max() + 1)

        traj = self.get_traj(i)
        height = self.estimate_heightmap(points, fill_value=0.)

        return cloud, traj, height

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.ids)


class RigidDEMPathData(DEMPathData):
    def __init__(self, path, cfg=Config()):
        super(RigidDEMPathData, self).__init__(path, cfg)

    def __getitem__(self, i, visualize=False):
        cloud = self.get_cloud(i)
        color = self.get_cloud_color(i)

        # merge cloud and colors
        cloud = merge_arrays([cloud, color], flatten=True, usemask=False)

        traj = self.get_traj(i)
        height = self.get_optimized_terrain(i)
        H, W = height.shape
        h, w = 2 * self.cfg.d_max // self.cfg.grid_res, 2 * self.cfg.d_max // self.cfg.grid_res
        # select only the h x w area from the center of the height map
        height = height[int(H // 2 - h // 2):int(H // 2 + h // 2),
                        int(W // 2 - w // 2):int(W // 2 + w // 2)]

        n = int(2 * self.cfg.d_max / self.cfg.grid_res)
        xi = np.linspace(-self.cfg.d_max, self.cfg.d_max, n)
        yi = np.linspace(-self.cfg.d_max, self.cfg.d_max, n)
        x_grid, y_grid = np.meshgrid(xi, yi)
        terrain = {
            'x': x_grid,
            'y': y_grid,
            'z': height
        }

        return cloud, traj, terrain


class MonoDEMData(DEMPathData):
    """
    A dataset for monocular traversability map estimation.

    A sample of the dataset contains:
    - image from a front camera (3 x H x W)
    - traversed height map from a front camera (1 x H x W)
    - lidar height map from a front camera (1 x H x W)
    - weights for traversed height map from a front camera (1 x H x W)
    - weights for lidar height map from a front camera (1 x H x W)
    """

    def __init__(self,
                 path,
                 cameras=None,
                 is_train=False,
                 random_camera_selection_prob=0.2,
                 cfg=Config()):
        super(MonoDEMData, self).__init__(path, cfg)
        self.img_size = cfg.img_size
        self.random_camera_selection_prob = random_camera_selection_prob
        self.is_train = is_train

        if cameras is None:
            cams_yaml = os.listdir(os.path.join(self.path, 'calibration/cameras'))
            cams = [cam.replace('.yaml', '') for cam in cams_yaml]
            if 'camera_up' in cams:
                cams.remove('camera_up')
            self.cameras = sorted(cams)
        else:
            self.cameras = cameras

        self.img_augs = A.Compose([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, always_apply=False, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.Blur(blur_limit=7, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.MotionBlur(blur_limit=7, p=0.5),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), p=0.5),
            # A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.5),
            A.RandomSunFlare(src_radius=100, num_flare_circles_lower=1, num_flare_circles_upper=2, p=0.5),
            # A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=0.5),
            A.RandomToneCurve(scale=0.1, p=0.5),
        ]) if self.is_train else None

    def get_image(self, i, cam, undistort=False):
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

    def resize_crop_img(self, img_raw):
        # resize image
        H_raw, W_raw = img_raw.shape[:2]
        h, w = self.img_size
        img = cv2.resize(img_raw, (int(h / H_raw * W_raw), h))
        # crop image
        H, W = img.shape[:2]
        img = img[H - h:H, W // 2 - w // 2: W // 2 + w // 2]
        return img

    def preprocess_img(self, img_raw):
        img = self.resize_crop_img(img_raw)
        if self.is_train:
            img = self.img_augs(image=img)['image']
        # img = standardize_img(img)
        return img

    def __getitem__(self, i, visualize=False):
        camera = 'camera_fisheye_front' if 'marv' in self.path else 'camera_front'
        # randomly choose a camera other than front camera
        if np.random.random() < self.random_camera_selection_prob and len(self.cameras) > 1 and self.is_train:
            cameras = self.cameras.copy()
            cameras.remove(camera)
            camera = np.random.choice(cameras)

        lidar = 'os_sensor'

        traj = self.get_traj(i)

        points_raw = position(self.get_raw_cloud(i))
        points = position(self.get_cloud(i))
        poses = traj['poses']

        img_front, K = self.get_image(i, camera.split('_')[-1])

        if visualize:
            # find transformation between camera and lidar
            lidar_to_camera = self.calib['transformations']['T_%s__%s' % (lidar, camera)]['data']
            lidar_to_camera = np.asarray(lidar_to_camera, dtype=float).reshape((4, 4))

            # transform point points to camera frame
            points_cam = transform_cloud(points_raw, lidar_to_camera)

            # project points to image
            points_fov, colors_view, fov_mask = project_cloud_to_image(points_cam, img_front, K, return_mask=True, debug=False)

            # set colors from a particular camera viewpoint
            colors = np.zeros_like(points_raw)
            colors[fov_mask] = colors_view[fov_mask]

        # square defining observation area on the ground
        square = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]]) * self.cfg.d_max / 2
        if 'front' in camera:
            offset = np.asarray([self.cfg.d_max / 2, 0, 0])
        elif 'left' in camera:
            offset = np.asarray([0, self.cfg.d_max / 2, 0])
        elif 'right' in camera:
            offset = np.asarray([0, -self.cfg.d_max / 2, 0])
        elif 'rear' in camera:
            offset = np.asarray([-self.cfg.d_max / 2, 0, 0])
        else:
            offset = np.asarray([0, 0, 0])
        square = square + offset  # move to robot frame

        # height map from point cloud (!!! assumes points are in robot frame)
        interpolation = self.cfg.hm_interp_method if self.cfg.hm_interp_method is not None else 'no_interp'
        dir_path = os.path.join(self.path, 'terrain', 'estimated', interpolation)
        # if height map was estimated before - load it
        if False and os.path.exists(os.path.join(dir_path, '%s.npy' % self.ids[i])):
            # print('Loading height map from file...')
            heightmap = np.load(os.path.join(dir_path, '%s.npy' % self.ids[i]))
        # otherwise - estimate it
        else:
            # print('Estimating height map...')
            heightmap = self.estimate_heightmap(points, fill_value=poses[0][2, 3])
            # save height map as numpy array
            result = np.zeros((heightmap['z'].shape[0], heightmap['z'].shape[1]), dtype=[(key, np.float64) for key in heightmap.keys()])
            for key in heightmap.keys():
                result[key] = heightmap[key]
            os.makedirs(dir_path, exist_ok=True)
            np.save(os.path.join(dir_path, '%s.npy' % self.ids[i]), result)
        # estimated height map
        height_est = heightmap['z']

        # optimized height map
        height_traj = self.get_optimized_terrain(i)

        # crop height map to observation area defined by square grid
        h, w = height_est.shape
        square_grid = square[:, :2] / self.cfg.grid_res + np.asarray([w / 2, h / 2])
        height_est_cam = height_est[int(square_grid[0, 1]):int(square_grid[2, 1]),
                                    int(square_grid[0, 0]):int(square_grid[2, 0])]
        height_traj_cam = height_traj[int(square_grid[0, 1]):int(square_grid[2, 1]),
                                    int(square_grid[0, 0]):int(square_grid[2, 0])]
        # poses in grid coordinates
        poses_grid = poses[:, :2, 3] / self.cfg.grid_res + np.asarray([w / 2, h / 2])
        # crop poses to observation area defined by square grid
        poses_grid_cam = poses_grid[(poses_grid[:, 0] > square_grid[0, 0]) & (poses_grid[:, 0] < square_grid[2, 0]) &
                                    (poses_grid[:, 1] > square_grid[0, 1]) & (poses_grid[:, 1] < square_grid[2, 1])]
        poses_grid_cam -= np.asarray([square_grid[0, 0], square_grid[0, 1]])

        # visited by poses dilated height map area mask
        H, W = height_traj_cam.shape
        kernel = np.ones((3, 3), dtype=np.uint8)
        weights_traj_cam = np.zeros((H, W), dtype=np.uint8)
        poses_grid_cam = poses_grid_cam.astype(np.uint32)
        weights_traj_cam[poses_grid_cam[:, 1], poses_grid_cam[:, 0]] = 1
        weights_traj_cam = cv2.dilate(weights_traj_cam, kernel, iterations=5)
        weights_traj_cam = weights_traj_cam.astype(bool)

        # circle mask: all points within a circle of radius 1 m are valid
        x_grid = np.arange(0, self.cfg.d_max, self.cfg.grid_res)
        y_grid = np.arange(-self.cfg.d_max / 2., self.cfg.d_max / 2., self.cfg.grid_res)
        x_grid, y_grid = np.meshgrid(x_grid, y_grid)
        # distances from the center
        radius = self.cfg.d_max / 2.
        dist = np.sqrt(x_grid ** 2 + y_grid ** 2)
        # gaussian mask
        weights_est_cam = np.exp(-dist ** 2 / (2. * radius ** 2))

        # rotate height maps and poses depending on camera orientation
        if 'left' in camera:
            height_est_cam = np.rot90(height_est_cam, 1)
            height_traj_cam = np.rot90(height_traj_cam, 1)
            weights_traj_cam = np.rot90(weights_traj_cam, 1)
        elif 'right' in camera:
            height_est_cam = np.rot90(height_est_cam, -1)
            height_traj_cam = np.rot90(height_traj_cam, -1)
            weights_traj_cam = np.rot90(weights_traj_cam, -1)
        elif 'rear' in camera:
            height_est_cam = np.rot90(height_est_cam, 2)
            height_traj_cam = np.rot90(height_traj_cam, 2)
            weights_traj_cam = np.rot90(weights_traj_cam, 2)

        # rotate heightmaps to have robot position at the bottom
        height_traj_cam = np.rot90(height_traj_cam, axes=(0, 1))
        height_est_cam = np.rot90(height_est_cam, axes=(0, 1))
        weights_traj_cam = np.rot90(weights_traj_cam, axes=(0, 1))
        weights_est_cam = np.rot90(weights_est_cam, axes=(0, 1))
        # flip heightmaps to have robot position at the bottom
        # we do copy, because of this issue:
        # https://stackoverflow.com/questions/72550211/valueerror-at-least-one-stride-in-the-given-numpy-array-is-negative-and-tensor
        height_traj_cam = np.fliplr(height_traj_cam).copy()
        height_est_cam = np.fliplr(height_est_cam).copy()
        weights_traj_cam = np.fliplr(weights_traj_cam).copy()
        weights_est_cam = np.fliplr(weights_est_cam).copy()
        
        if visualize:
            # draw point cloud with mayavi
            color_n = np.arange(len(points))
            lut = np.zeros((len(color_n), 4))
            lut[:, :3] = colors
            lut[:, 3] = 255

            mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
            # draw point cloud
            p3d = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color_n, mode='point', opacity=0.5, scale_factor=0.1)
            p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut)
            p3d.module_manager.scalar_lut_manager.lut.table = lut
            # draw poses
            mlab.points3d(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], color=(1, 0, 0), scale_factor=0.4)
            draw_coord_frame(poses[0], scale=2.)
            draw_coord_frames(poses, scale=0.5)
            mlab.plot3d(square[:, 0], square[:, 1], square[:, 2], color=(0, 1, 0), line_width=5, tube_radius=0.1)
            mlab.view(azimuth=0, elevation=0, distance=2 * self.cfg.d_max)

            plt.figure(figsize=(20, 20))
            plt.subplot(241)
            plt.title('RGB image')
            plt.imshow(img_front)
            plt.axis('off')

            plt.subplot(242)
            plt.title('Estimated heightmap')
            plt.imshow(height_est, cmap='jet', alpha=0.8, origin='lower')
            # plot trajectory
            plt.plot(poses_grid[:, 0], poses_grid[:, 1], 'ro', markersize=2)
            # plot square
            plt.plot(square_grid[:, 0], square_grid[:, 1], 'y--', linewidth=2)
            # plt.grid()
            # plt.colorbar()

            plt.subplot(243)
            plt.title('Estimated heightmap in camera frame')
            plt.imshow(height_est_cam, cmap='jet', alpha=1.)
            # plt.imshow(weights_est_cam, cmap='gray', alpha=0.5)
            plt.colorbar()

            plt.subplot(244)
            plt.title('Estimated heightmap weights')
            plt.imshow(weights_est_cam, cmap='gray', alpha=1.)

            plt.subplot(245)
            # plot point cloud
            points_filtered = filter_range(points, self.cfg.d_min, self.cfg.d_max)
            points_filtered = filter_grid(points_filtered, 0.2)
            points_grid = points_filtered[:, :2] / self.cfg.grid_res + np.asarray([w / 2, h / 2])
            plt.plot(points_grid[:, 0], points_grid[:, 1], 'k.', markersize=1, alpha=0.5)
            plt.axis('equal')

            plt.subplot(246)
            plt.title('Optimized heightmap')
            plt.imshow(height_traj, cmap='jet', alpha=0.8, origin='lower')
            plt.plot(poses_grid[:, 0], poses_grid[:, 1], 'ro', markersize=2)
            plt.plot(square_grid[:, 0], square_grid[:, 1], 'y--', linewidth=2)
            # plt.colorbar()

            plt.subplot(247)
            plt.title('Optimized heightmap in camera frame')
            plt.imshow(height_traj_cam, cmap='jet', alpha=1.)
            plt.colorbar()

            plt.subplot(248)
            plt.title('Optimized heightmap weights')
            plt.imshow(weights_traj_cam, cmap='gray', alpha=1.)

            # mlab.show()
            plt.show()

        # resize and normalize image
        img_front = self.preprocess_img(img_front)

        # flip image and heightmaps from left to right with 50% probability
        if self.is_train and np.random.random() > 0.5:
            img_front = np.fliplr(img_front).copy()
            height_traj_cam = np.fliplr(height_traj_cam).copy()
            height_est_cam = np.fliplr(height_est_cam).copy()
            weights_traj_cam = np.fliplr(weights_traj_cam).copy()
            weights_est_cam = np.fliplr(weights_est_cam).copy()

        # convert to CHW format
        img_front_CHW = img_front.transpose((2, 0, 1))

        return img_front_CHW, height_traj_cam[None], height_est_cam[None], weights_traj_cam[None], weights_est_cam[None]


class OmniDEMData(MonoDEMData):
    def __init__(self,
                 path,
                 data_aug_conf,
                 is_train=True,
                 cfg=Config()):
        super(OmniDEMData, self).__init__(path, is_train=is_train, cfg=cfg)

        self.data_aug_conf = data_aug_conf

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, i, normalize=True):
        imgs = []
        rots = []
        trans = []
        post_rots = []
        post_trans = []
        intrins = []

        # permute cameras
        cameras = self.cameras.copy()
        # if self.is_train:
        #     np.random.shuffle(cameras)

        for cam in cameras:
            img, K = self.get_image(i, cam, undistort=False)

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
            if f'T_base_link__{cam}' in self.calib['transformations'].keys():
                T_cam_robot = self.calib['transformations'][f'T_base_link__{cam}']['data']
                T_cam_robot = np.asarray(T_cam_robot, dtype=np.float32).reshape((4, 4))
                T_robot_cam = np.linalg.inv(T_cam_robot)
            else:
                T_lidar_cam = self.calib['transformations']['T_os_sensor__%s' % cam]['data']
                T_lidar_cam = np.asarray(T_lidar_cam, dtype=np.float32).reshape((4, 4))
                T_cam_lidar = np.linalg.inv(T_lidar_cam)
                T_robot_lidar = self.calib['transformations']['T_base_link__os_sensor']['data']
                T_robot_lidar = np.asarray(T_robot_lidar, dtype=np.float32).reshape((4, 4))
                T_robot_cam = T_robot_lidar @ T_cam_lidar

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

    def get_lidar_height_map(self, i, cached=True):
        # height map from point cloud (!!! assumes points are in robot frame)
        interpolation = self.cfg.hm_interp_method if self.cfg.hm_interp_method is not None else 'no_interp'
        dir_path = os.path.join(self.path, 'terrain', 'estimated', interpolation)
        # if height map was estimated before - load it
        if cached and os.path.exists(os.path.join(dir_path, '%s.npy' % self.ids[i])):
            # print('Loading height map from file...')
            xyz_mask = np.load(os.path.join(dir_path, '%s.npy' % self.ids[i]))
        # otherwise - estimate it
        else:
            # print('Estimating and saving height map...')
            cloud = self.get_cloud(i)
            points = position(cloud)
            xyz_mask = estimate_heightmap(points,
                                          d_min=self.cfg.d_min, d_max=self.cfg.d_max,
                                          grid_res=self.cfg.grid_res, h_max=self.cfg.h_max,
                                          hm_interp_method=self.hm_interp_method)
            # save height map as numpy array
            result = np.zeros((xyz_mask['z'].shape[0], xyz_mask['z'].shape[1]),
                              dtype=[(key, np.float32) for key in xyz_mask.keys()])
            for key in xyz_mask.keys():
                result[key] = xyz_mask[key]
            os.makedirs(dir_path, exist_ok=True)
            np.save(os.path.join(dir_path, '%s.npy' % self.ids[i]), result)

        heightmap = torch.stack([torch.as_tensor(xyz_mask[i]) for i in ['z', 'mask']])
        heightmap = torch.as_tensor(heightmap, dtype=torch.float32)

        return heightmap

    def global_hm_cloud(self, vis=False):
        poses = self.poses
        # create global heightmap cloud
        global_hm_cloud = []
        for i in tqdm(range(len(self))):
            hm = self.get_lidar_height_map(i)
            hm_cloud = hm_to_cloud(hm[0], self.cfg, mask=hm[1])
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

    def __getitem__(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(i)
        height = self.get_lidar_height_map(i)
        return imgs, rots, trans, intrins, post_rots, post_trans, height


class OmniDEMDataVis(OmniDEMData):
    def __init__(self,
                 path,
                 data_aug_conf,
                 is_train=True,
                 cfg=Config()
                 ):
        super(OmniDEMDataVis, self).__init__(path, data_aug_conf, is_train=is_train, cfg=cfg)

    def __getitem__(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(i)
        height = self.get_lidar_height_map(i)
        lidar_pts = torch.as_tensor(position(self.get_cloud(i))).T
        return imgs, rots, trans, intrins, post_rots, post_trans, height, lidar_pts


class OmniRigidDEMData(OmniDEMData):
    def __init__(self,
                 path,
                 data_aug_conf,
                 is_train=True,
                 cfg=Config()
                 ):
        super(OmniRigidDEMData, self).__init__(path, data_aug_conf, is_train=is_train, cfg=cfg)

    def get_traj_height_map(self, i, method='footprint'):
        assert method in ['optimized', 'footprint']
        if method == 'optimized':
            height = self.get_optimized_terrain(i)
            # Optimized height map shape is 256 x 256. We need to crop it to 128 x 128
            H, W = height.shape
            h, w = int(2 * self.cfg.d_max // self.cfg.grid_res), int(2 * self.cfg.d_max // self.cfg.grid_res)
            # select only the h x w area from the center of the height map
            height = height[int(H // 2 - h // 2):int(H // 2 + h // 2),
                            int(W // 2 - w // 2):int(W // 2 + w // 2)]

            # poses in grid coordinates
            poses = self.get_traj(i)['poses']
            poses_grid = poses[:, :2, 3] / self.cfg.grid_res + np.asarray([w / 2, h / 2])
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
            traj_points = self.estimated_footprint_traj_points(i)
            traj_hm = self.estimate_heightmap(traj_points, robot_size=None)
            height = traj_hm['z']
            mask = traj_hm['mask']

        height = torch.from_numpy(height)
        mask = torch.from_numpy(mask)
        heightmap = torch.stack([height, mask])
        return heightmap

    def __getitem__(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(i)
        height = self.get_traj_height_map(i)
        return imgs, rots, trans, intrins, post_rots, post_trans, height


class DepthDEMData(OmniDEMData):
    def __init__(self,
                 path,
                 data_aug_conf,
                 is_train=True,
                 cfg=Config()
                 ):
        super(DepthDEMData, self).__init__(path, data_aug_conf, is_train=is_train, cfg=cfg)

    def get_raw_image(self, i, camera='realsense_front'):
        if camera in ['front', 'rear', 'left', 'right']:
            prefix = 'realsense_'
            camera = prefix + camera
        ind = self.ids[i]
        img_path = os.path.join(self.path, 'depths/visuals', '%s_%s.png' % (ind, camera))
        assert os.path.exists(img_path), f'Image path {img_path} does not exist'
        img = Image.open(img_path)
        img = np.asarray(img)
        return img

    def get_cloud(self, i):
        cloud = self.get_raw_cloud(i)
        cloud = filter_grid(cloud, self.cfg.grid_res)
        # cloud = filter_range(cloud, self.cfg.d_min, self.cfg.d_max)
        # move points to robot frame
        Tr = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr = np.asarray(Tr, dtype=float).reshape((4, 4))
        Tr = np.linalg.inv(Tr)
        cloud = transform_cloud(cloud, Tr)
        return cloud

    def __getitem__(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(i, normalize=False)
        height = self.get_lidar_height_map(i)
        return imgs, rots, trans, intrins, post_rots, post_trans, height


class DepthDEMDataVis(DepthDEMData):
    def __init__(self,
                    path,
                    data_aug_conf,
                    is_train=True,
                    cfg=Config()
                    ):
            super(DepthDEMDataVis, self).__init__(path, data_aug_conf, is_train=is_train, cfg=cfg)

    def __getitem__(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(i, normalize=False)
        height = self.get_lidar_height_map(i)
        lidar_pts = torch.as_tensor(position(self.get_cloud(i))).T
        return imgs, rots, trans, intrins, post_rots, post_trans, height, lidar_pts


class OmniRigidDEMDataVis(OmniRigidDEMData):
    def __init__(self,
                 path,
                 data_aug_conf,
                 is_train=True,
                 cfg=Config()
                 ):
        super(OmniRigidDEMDataVis, self).__init__(path, data_aug_conf, is_train=is_train, cfg=cfg)

    def __getitem__(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(i)
        height = self.get_traj_height_map(i)
        lidar_pts = torch.as_tensor(position(self.get_cloud(i))).T
        return imgs, rots, trans, intrins, post_rots, post_trans, height, lidar_pts


class TravData(OmniRigidDEMData):
    def __init__(self,
                 path,
                 data_aug_conf,
                 is_train=True,
                 cfg=Config()
                 ):
        super(TravData, self).__init__(path, data_aug_conf, is_train=is_train, cfg=cfg)

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(i)
        height_lidar = self.get_lidar_height_map(i)
        height_traj = self.get_traj_height_map(i)
        map_pose = torch.as_tensor(self.get_pose(i))
        sample = (imgs, rots, trans, intrins, post_rots, post_trans, height_lidar, height_traj, map_pose)
        return sample

    def __getitem__(self, i):
        if isinstance(i, (int, np.int64)):
            sample = self.get_sample(i)
            return sample

        ds = TravData(self.path, self.data_aug_conf, is_train=self.is_train, cfg=self.cfg)
        if isinstance(i, (list, tuple, np.ndarray)):
            ds.ids = [self.ids[k] for k in i]
            ds.poses = [self.poses[k] for k in i]
        else:
            assert isinstance(i, (slice, range))
            ds.ids = self.ids[i]
            ds.poses = self.poses[i]
        return ds

class TravDataVis(TravData):
    def __init__(self,
                 path,
                 data_aug_conf,
                 is_train=True,
                 cfg=Config()
                 ):
          super(TravDataVis, self).__init__(path, data_aug_conf, is_train=is_train, cfg=cfg)

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(i)
        height_lidar = self.get_lidar_height_map(i)
        height_traj = self.get_traj_height_map(i)
        map_pose = torch.as_tensor(self.get_pose(i))
        lidar_pts = torch.as_tensor(position(self.get_cloud(i))).T
        sample = (imgs, rots, trans, intrins, post_rots, post_trans, height_lidar, height_traj, map_pose, lidar_pts)
        return sample

    def __getitem__(self, i):
        if isinstance(i, (int, np.int64)):
            sample = self.get_sample(i)
            return sample

        ds = TravDataVis(self.path, self.data_aug_conf, is_train=self.is_train, cfg=self.cfg)
        if isinstance(i, (list, tuple, np.ndarray)):
            ds.ids = [self.ids[k] for k in i]
            ds.poses = [self.poses[k] for k in i]
        else:
            assert isinstance(i, (slice, range))
            ds.ids = self.ids[i]
            ds.poses = self.poses[i]
        return ds


def heightmap_demo():
    from ..vis import show_cloud_plt
    from ..cloudproc import filter_grid, filter_range
    import matplotlib.pyplot as plt

    path = seq_paths[0]
    assert os.path.exists(path)

    cfg = Config()
    # ds = OptDEMTrajData(path, cfg=cfg)
    ds = DEMPathData(path, cfg=cfg)

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

    for path in seq_paths:
        assert os.path.exists(path)

        cfg = Config()
        ds = DEMPathData(path, cfg=cfg)

        robot_pose = np.eye(4)
        robot_frame = 'base_link'
        lidar_frame = 'os_sensor'

        Tr_robot_lidar = ds.calib['transformations'][f'T_{robot_frame}__{lidar_frame}']['data']
        Tr_robot_lidar = np.asarray(Tr_robot_lidar, dtype=float).reshape((4, 4))

        if 'robingas' in path:
            if 'marv' in path:
                camera_frames = ['camera_left', 'camera_right', 'camera_up', 'camera_fisheye_front', 'camera_fisheye_rear']
            else:
                camera_frames = ['camera_left', 'camera_right', 'camera_front', 'camera_rear']
        elif 'sim' in path:
            camera_frames = ['realsense_left', 'realsense_right', 'realsense_front', 'realsense_rear']
        else:
            camera_frames = ['ids_camera']

        cam_poses = []
        for frame in camera_frames:
            Tr_lidar_cam = ds.calib['transformations'][f'T_{lidar_frame}__{frame}']['data']
            Tr_lidar_cam = np.asarray(Tr_lidar_cam, dtype=float).reshape((4, 4))
            Tr_cam_lidar= np.linalg.inv(Tr_lidar_cam)
            Tr_robot_cam = Tr_robot_lidar @ Tr_cam_lidar
            cam_poses.append(Tr_robot_cam[np.newaxis])
        cam_poses = np.concatenate(cam_poses, axis=0)

        # draw coordinate frames
        mlab.figure(size=(800, 800))
        draw_coord_frame(robot_pose, scale=0.5)
        draw_coord_frame(Tr_robot_lidar, scale=0.3)
        draw_coord_frames(cam_poses, scale=0.1)
        mlab.show()


def vis_rgb_cloud():
    from ..vis import show_cloud
    from ..utils import normalize

    for path in seq_paths:
        assert os.path.exists(path)

        cfg = Config()
        ds = DEMPathData(path, cfg=cfg)

        i = np.random.choice(range(len(ds)))
        # i = 10
        cloud = ds.get_cloud(i)
        colors = ds.get_cloud_color(i)
        if colors is None:
            colors = np.zeros_like(cloud)

        # poses
        traj = ds.get_traj(i)
        poses = traj['poses']

        # images
        img_front = ds.get_raw_image(i, 'front')
        img_rear = ds.get_raw_image(i, 'rear')
        img_left = ds.get_raw_image(i, 'left')
        img_right = ds.get_raw_image(i, 'right')

        # colored point cloud
        points = position(cloud)
        rgb = color(colors)
        rgb = normalize(rgb)

        # show images
        plt.figure(figsize=(12, 12))

        plt.subplot(332)
        plt.title('Front camera')
        plt.imshow(img_front)
        plt.axis('off')

        plt.subplot(338)
        plt.title('Rear camera')
        plt.imshow(img_rear)
        plt.axis('off')

        plt.subplot(334)
        plt.title('Left camera')
        plt.imshow(img_left)
        plt.axis('off')

        plt.subplot(336)
        plt.title('Right camera')
        plt.imshow(img_right)
        plt.axis('off')

        # show point cloud
        plt.subplot(335)
        points_vis = filter_range(points, cfg.d_min, cfg.d_max)
        points_vis = filter_grid(points_vis, 0.2)
        plt.plot(points_vis[:, 0], points_vis[:, 1], 'k.', markersize=0.5)
        # plot poses
        plt.plot(poses[:, 0, 3], poses[:, 1, 3], 'ro', markersize=4)
        plt.axis('equal')
        plt.grid()

        plt.show()

        show_cloud(points, rgb)


def traversed_height_map():
    path = np.random.choice(seq_paths)
    assert os.path.exists(path)

    cfg = Config()
    cfg.from_yaml(os.path.join(path, 'terrain', 'train_log', 'cfg.yaml'))
    # cfg.d_min = 1.

    ds = DEMPathData(path, cfg=cfg)
    i = np.random.choice(range(len(ds)))

    # trajectory poses
    poses = ds.get_traj(i)['poses']
    # point cloud
    cloud = ds.get_cloud(i)
    points = position(cloud)

    img = ds.get_raw_image(i)

    # height map: estimated from point cloud
    heightmap = ds.estimate_heightmap(points)
    height_est = heightmap['z']
    x_grid, y_grid = heightmap['x'], heightmap['y']
    # height map: optimized from robot-terrain interaction model
    height = ds.get_optimized_terrain(i)

    plt.figure(figsize=(12, 12))
    h, w = height_est.shape
    xy_grid = poses[:, :2, 3] / cfg.grid_res + np.array([w / 2, h / 2])
    plt.subplot(131)
    plt.imshow(height_est)
    plt.plot(xy_grid[:, 0], xy_grid[:, 1], 'rx', markersize=4)
    plt.subplot(132)
    plt.imshow(height)
    plt.plot(xy_grid[:, 0], xy_grid[:, 1], 'rx', markersize=4)
    plt.subplot(133)
    plt.imshow(img)
    plt.show()


def vis_train_sample():
    cfg = Config()
    path = np.random.choice(seq_paths)
    cfg.from_yaml(os.path.join(path, 'terrain', 'train_log', 'cfg.yaml'))

    ds = MonoDEMData(path=path, cfg=cfg)
    i = np.random.choice(range(len(ds)))
    # i = 0
    print(f'Visualizing sample {i}...')
    img, height_traj, height_est, weights_traj, weights_est = ds.__getitem__(i, visualize=True)
    # img, height_traj, height_est, weights_traj = ds[i]
    img = img.transpose(1, 2, 0)

    plt.figure(figsize=(20, 7))
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.title('Height Label')
    plt.imshow(height_traj.squeeze(), cmap='jet')
    plt.imshow(weights_traj.squeeze(), alpha=0.5, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Height Regularization')
    plt.imshow(height_est.squeeze(), cmap='jet')
    plt.show()


def vis_hm_weights():
    cfg = Config()

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
    cfg = Config()
    cfg.grid_res = 0.1
    cfg.d_max = 12.8
    cfg.d_min = 1.
    cfg.h_max = 1.
    # cfg.hm_interp_method = None
    cfg.hm_interp_method = 'nearest'

    path = np.random.choice(seq_paths)
    ds = DEMPathData(path=path, cfg=cfg)

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


def vis_img_augs():
    path = np.random.choice(seq_paths)
    cfg = Config()
    ds = MonoDEMData(path=path,
                     is_train=True,
                     cfg=cfg)
    i = 0
    # img_raw = ds.get_image(i, 'front')
    img_raw, _ = ds.get_image(i, 'front')
    # ds.img_augs = A.Compose([
    #     # A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, always_apply=True),
    #     # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, always_apply=True),
    #     # A.RandomGamma(gamma_limit=(80, 120), always_apply=True),
    #     # A.Blur(blur_limit=7, always_apply=True),
    #     # A.GaussNoise(var_limit=(10, 50), always_apply=True),
    #     # A.MotionBlur(blur_limit=7, always_apply=True),
    #     # A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200)),
    #     # A.RandomSunFlare(src_radius=100, num_flare_circles_lower=1, num_flare_circles_upper=2),
    #     # A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5),
    #     A.RandomToneCurve(scale=0.1, always_apply=True),
    # ])
    # img = ds.img_augs(image=img_raw)["image"]
    img, height_traj, height_est, weights_traj, weights_est = ds[i]
    img = img.transpose(1, 2, 0)

    plt.figure(figsize=(20, 7))
    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow(img_raw)
    plt.subplot(1, 2, 2)
    plt.title('Augmented Image')
    plt.imshow(img)
    plt.show()


def global_cloud_demo():
    for path in seq_paths:
        ds = DEMPathData(path=path)
        ds.global_cloud(vis=True)


def explore_data(path, grid_conf, data_aug_conf, cfg, modelf=None,
                 sample_range='random', save=False, is_train=False):
    assert os.path.exists(path)

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    if modelf is not None:
        model.load_state_dict(torch.load(modelf))
        print('Loaded LSS model from', modelf)
        model.eval()

    ds = TravDataVis(path, is_train=is_train, data_aug_conf=data_aug_conf, cfg=cfg)

    H, W = data_aug_conf['H'], data_aug_conf['W']
    cams = data_aug_conf['cams']
    rat = H / W
    val = 10.1

    if sample_range == 'random':
        sample_range = [np.random.choice(range(len(ds)))]
    elif sample_range == 'all':
        sample_range = tqdm(range(len(ds)), total=len(ds))
    else:
        assert isinstance(sample_range, list) or isinstance(sample_range, np.ndarray) or isinstance(sample_range, range)

    for sample_i in sample_range:
        fig = plt.figure(figsize=(val + val/3*2*rat*3, val/3*2*rat))
        gs = mpl.gridspec.GridSpec(2, 5, width_ratios=(1, 1, 2 * rat, 2 * rat, 2 * rat))
        gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

        sample = ds[sample_i]
        sample = [s[np.newaxis] for s in sample]
        imgs, rots, trans, intrins, post_rots, post_trans, hm_lidar, hm_traj, map_pose, pts = sample
        if modelf is not None:
            with torch.no_grad():
                inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
                inputs = [torch.as_tensor(i, dtype=torch.float32) for i in inputs]
                hm_lidar = model(*inputs)

        img_pts = model.get_geometry(rots, trans, intrins, post_rots, post_trans)

        for si in range(imgs.shape[0]):
            plt.clf()
            final_ax = plt.subplot(gs[:, 4:5])
            for imgi, img in enumerate(imgs[si]):
                ego_pts = ego_to_cam(pts[si], rots[si, imgi], trans[si, imgi], intrins[si, imgi])
                mask = get_only_in_img_mask(ego_pts, H, W)
                plot_pts = post_rots[si, imgi].matmul(ego_pts) + post_trans[si, imgi].unsqueeze(1)

                ax = plt.subplot(gs[imgi // 2, imgi % 2])
                showimg = denormalize_img(img)

                plt.imshow(showimg)
                plt.scatter(plot_pts[0, mask], plot_pts[1, mask], c=ego_pts[2, mask], s=1, alpha=0.1, cmap='jet')
                plt.axis('off')
                # camera name as text on image
                plt.text(0.5, 0.9, cams[imgi].replace('_', ' '),
                         horizontalalignment='center', verticalalignment='top',
                         transform=ax.transAxes, fontsize=10)

                plt.sca(final_ax)
                plt.plot(img_pts[si, imgi, :, :, :, 0].view(-1), img_pts[si, imgi, :, :, :, 1].view(-1), '.',
                         label=cams[imgi].replace('_', ' '))

            plt.legend(loc='upper right')
            final_ax.set_aspect('equal')
            plt.xlim((-cfg.d_max, cfg.d_max))
            plt.ylim((-cfg.d_max, cfg.d_max))

            ax = plt.subplot(gs[:, 2:3])
            plt.imshow(hm_lidar[si][0].T, origin='lower', cmap='jet', vmin=-1., vmax=1.)
            plt.axis('off')
            plt.colorbar()

            ax = plt.subplot(gs[:, 3:4])
            plt.imshow(hm_traj[si][0].T, origin='lower', cmap='jet', vmin=-1., vmax=1.)
            plt.axis('off')
            plt.colorbar()

            if save:
                save_dir = os.path.join(path, 'visuals')
                os.makedirs(save_dir, exist_ok=True)
                imname = f'{ds.ids[sample_i]}.jpg'
                imname = os.path.join(save_dir, imname)
                # print('saving', imname)
                plt.savefig(imname)
                plt.close(fig)
            else:
                plt.show()


def geometric_traversed_heightmap():
    import open3d as o3d

    path = seq_paths[0]
    assert os.path.exists(path)

    cfg = Config()
    ds = DEMPathData(path, cfg=cfg)

    i = np.random.choice(range(len(ds)))
    sample = ds[i]
    cloud, traj, height = sample
    points = position(cloud)

    traj_points = ds.estimated_footprint_traj_points(i)

    lidar_height = ds.estimate_heightmap(points)['z']
    traj_hm = ds.estimate_heightmap(traj_points, robot_size=None)
    traj_height = traj_hm['z']
    traj_mask = traj_hm['mask']
    print('lidar_height', lidar_height.shape)
    print('traj_height', traj_height.shape)

    plt.figure()
    plt.subplot(131)
    plt.imshow(lidar_height, cmap='jet', vmin=-0.5, vmax=0.5)
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(traj_height, cmap='jet', vmin=-0.5, vmax=0.5)
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(traj_mask, cmap='gray')
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
    vis_rgb_cloud()
    traversed_height_map()
    vis_train_sample()
    vis_hm_weights()
    vis_estimated_height_map()
    vis_img_augs()
    global_cloud_demo()
    geometric_traversed_heightmap()


if __name__ == '__main__':
    main()
