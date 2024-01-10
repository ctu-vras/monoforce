import os
import matplotlib as mpl
import numpy as np
import torch
from torch.utils.data import Dataset
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured, merge_arrays
from matplotlib import cm, pyplot as plt
from mayavi import mlab
from ..models.lss.model import compile_model
from ..config import Config
from ..transformations import transform_cloud
from ..cloudproc import position, estimate_heightmap, color
from ..cloudproc import filter_grid, filter_range
from ..imgproc import undistort_image, project_cloud_to_image, standardize_img, destandardize_img
from .augmentations import horizontal_shift
from ..vis import show_cloud, draw_coord_frame, draw_coord_frames, set_axes_equal
from ..utils import normalize
from .utils import load_cam_calib
import cv2
import albumentations as A
from ..models.lss.tools import img_transform, ego_to_cam, get_only_in_img_mask
from PIL import Image
from tqdm import tqdm
try:
    import matplotlib
    matplotlib.use('QtAgg')
except:
    print('Could not set matplotlib backend to QtAgg')
    pass


__all__ = [
    'SegmentationDataset',
    'DEMTrajData',
    'OptDEMTrajData',
    'MonoDEMData',
    'OmniDEMData',
    'OmniDEMDataVis',
    'OmniOptDEMData',
    'OmniOptDEMDataVis',
    'seq_paths',
    'sim_seq_paths',
]

IGNORE_LABEL = 255
data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))

seq_paths = [
        os.path.join(data_dir, 'robingas/data/22-10-27-unhost-final-demo/husky_2022-10-27-15-33-57_trav/'),
        os.path.join(data_dir, 'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-10-33-15_trav/'),
        os.path.join(data_dir, 'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-15-01-44_trav/'),
        os.path.join(data_dir, 'robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-16-37-03_trav/'),
        os.path.join(data_dir, 'robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'),
]
seq_paths = [os.path.normpath(path) for path in seq_paths]

sim_seq_paths = [
        os.path.join(data_dir, 'lss_input/husky_emptyfarm_2024-01-03-13-36-25_trav'),
        os.path.join(data_dir, 'lss_input/husky_farmWith1CropRow_2024-01-03-13-52-36_trav'),
        os.path.join(data_dir, 'lss_input/husky_inspection_2024-01-03-14-06-53_trav'),
        os.path.join(data_dir, 'lss_input/husky_simcity_2024-01-03-13-55-37_trav'),
        os.path.join(data_dir, 'lss_input/husky_simcity_dynamic_2024-01-03-13-59-08_trav'),
        os.path.join(data_dir, 'lss_input/husky_simcity_2024-01-09-17-56-34_trav'),
        os.path.join(data_dir, 'lss_input/husky_simcity_2024-01-09-17-50-23_trav'),
        os.path.join(data_dir, 'lss_input/husky_emptyfarm_vegetation_2024-01-09-17-18-46_trav'),
]
sim_seq_paths = [os.path.normpath(path) for path in sim_seq_paths]


class SegmentationDataset(Dataset):
    """
    Class to wrap semi-supervised traversability data generated using lidar odometry and IMU.
    Please, have a look at the `save_clouds_and_trajectories_from_bag` script for data generation from bag file.

    A sample of the dataset contains:
    - depth projection of a point cloud (H x W)
    - semantic label projection of a point cloud (H x W)
    """

    def __init__(self, path, split='val'):
        super(Dataset, self).__init__()
        assert split in ['train', 'val']
        self.split = split
        self.path = path
        self.cloud_path = os.path.join(path, 'clouds')
        assert os.path.exists(self.cloud_path)
        self.traj_path = os.path.join(path, 'trajectories')
        assert os.path.exists(self.traj_path)
        self.ids = [f[:-4] for f in os.listdir(self.cloud_path)]
        self.proj_fov_up = 45
        self.proj_fov_down = -45
        self.proj_H = 128
        self.proj_W = 1024
        self.ignore_label = IGNORE_LABEL

    def range_projection(self, points, labels):
        """ Project a point cloud into a sphere.
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(points, 2, axis=1)

        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / (depth + 1e-8))

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]
        indices = indices[order]

        # assing to image
        proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_range[proj_y, proj_x] = depth

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices
        # only map colors to labels that exist
        mask = proj_idx >= 0

        # projection color with semantic labels
        proj_sem_label = np.full((self.proj_H, self.proj_W), self.ignore_label, dtype=np.float32)  # [H,W]  label
        proj_sem_label[mask] = labels[proj_idx[mask]]

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)
        proj_xyz[proj_y, proj_x] = points[order]

        return proj_range, proj_sem_label, proj_xyz

    def __getitem__(self, i, visualize=False):
        ind = self.ids[i]
        cloud = np.load(os.path.join(self.cloud_path, '%s.npz' % ind))['cloud']

        if cloud.ndim == 2:
            cloud = cloud.reshape((-1,))

        points = position(cloud)
        trav = np.asarray(cloud['traversability'], dtype=points.dtype)

        depth_proj, label_proj, points_proj = self.range_projection(points, trav)

        if self.split == 'train':
            # data augmentation: add rotation around vertical axis (Z)
            H, W = depth_proj.shape
            shift = np.random.choice(range(1, W))
            depth_proj = horizontal_shift(depth_proj, shift=shift)
            label_proj = horizontal_shift(label_proj, shift=shift)
            # point projected have shape (H, W, 3)
            points_proj_shifted = np.zeros_like(points_proj)
            points_proj_shifted[:, :shift, :] = points_proj[:, -shift:, :]
            points_proj_shifted[:, shift:, :] = points_proj[:, :-shift, :]
            points_proj = points_proj_shifted

        points_proj = points_proj.reshape((-1, 3))

        if visualize:
            valid = trav != self.ignore_label
            show_cloud(points_proj, label_proj.reshape(-1, ),
                       min=trav[valid].min(), max=trav[valid].max() + 1, colormap=cm.jet)

        return depth_proj[None], label_proj[None]

    def __len__(self):
        return len(self.ids)


class DEMTrajData(Dataset):
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
        self.ids = np.sort([f[:-4] for f in os.listdir(self.cloud_path)])
        self.cfg = cfg
        self.calib = load_cam_calib(calib_path=self.calib_path)
        self.hm_interp_method = self.cfg.hm_interp_method

    @staticmethod
    def pose2mat(pose):
        T = np.eye(4)
        T[:3, :4] = pose.reshape((3, 4))
        return T

    def get_poses(self):
        data = np.loadtxt(self.poses_path, delimiter=',', skiprows=1)
        stamps, Ts = data[:, 0], data[:, 1:13]
        poses = np.asarray([self.pose2mat(pose) for pose in Ts])
        # poses = {}
        # for i, stamp in enumerate(stamps):
        #     poses[stamp] = self.pose2mat(Ts[i])
        return poses

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
        Tr = np.asarray(Tr, dtype=float).reshape((4, 4))
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
        rgb = np.load(os.path.join(self.cloud_color_path, '%s.npz' % ind))['rgb']
        # convert to structured numpy array with 'r', 'g', 'b' fields
        color = unstructured_to_structured(rgb, names=['r', 'g', 'b'])
        return color

    def get_image(self, i, camera='front'):
        if camera in ['front', 'rear', 'left', 'right', 'up']:
            prefix = 'camera_fisheye_' if 'marv' in self.path and camera in ['front', 'rear'] else 'camera_'
            camera = prefix + camera
        ind = self.ids[i]
        img_path = os.path.join(self.path, 'images', '%s_%s.png' % (ind, camera))
        assert os.path.exists(img_path), f'Image path {img_path} does not exist'
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image[..., ::-1]
        return image

    def get_optimized_terrain(self, i):
        ind = self.ids[i]
        terrain = np.load(os.path.join(self.path, 'terrain', '%s.npy' % ind))
        # TODO: transformation issue (need to add a transposition hack)
        return terrain.T

    def global_cloud(self, colorize=False, vis=False, step_size=1):
        poses = self.get_poses()

        # create global cloud
        for i in tqdm(range(len(self))):
            cloud = self.get_raw_cloud(i)
            if colorize:
                # cloud color
                color_struct = self.get_cloud_color(i)
                rgb = normalize(color(color_struct))
            T = poses[i]
            cloud = transform_cloud(cloud, T)
            points = position(cloud)
            if i == 0:
                global_cloud = points[::step_size]
                global_cloud_rgb = rgb[::step_size] if colorize else None
            else:
                global_cloud = np.vstack((global_cloud, points[::step_size]))
                global_cloud_rgb = np.vstack((global_cloud_rgb, rgb[::step_size])) if colorize else None

        if vis:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(global_cloud)
            if colorize:
                pcd.colors = o3d.utility.Vector3dVector(global_cloud_rgb)
            o3d.visualization.draw_geometries([pcd])

            # plt.figure(figsize=(10, 10))
            # # plot global cloud
            # plt.scatter(global_cloud[::100, 0], global_cloud[::100, 1], s=1, c='k')
            # # plot poses
            # plt.scatter(poses[:, 0, 3], poses[:, 1, 3], s=10, c='r')
            # plt.grid()
            # plt.axis('equal')
            # plt.show()

        return global_cloud

    def estimate_heightmap(self, points, fill_value=None, return_filtered_points=False):
        # estimate heightmap from point cloud
        height = estimate_heightmap(points, d_min=self.cfg.d_min, d_max=self.cfg.d_max,
                                    grid_res=self.cfg.grid_res, h_max=self.cfg.h_max,
                                    hm_interp_method=self.hm_interp_method,
                                    fill_value=fill_value, return_filtered_points=return_filtered_points)
        return height

    def __getitem__(self, i, visualize=False):
        cloud = self.get_cloud(i)
        points = position(cloud)
        color = self.get_cloud_color(i)
        # merge cloud and colors
        cloud = merge_arrays([cloud, color], flatten=True, usemask=False)

        if visualize:
            trav = np.asarray(cloud['traversability'], dtype=points.dtype)
            valid = trav != IGNORE_LABEL
            show_cloud(points, min=trav[valid].min(), max=trav[valid].max() + 1, colormap=cm.jet)

        traj = self.get_traj(i)
        height = self.estimate_heightmap(points, fill_value=0.)

        return cloud, traj, height

    def __len__(self):
        return len(self.ids)


class OptDEMTrajData(DEMTrajData):
    def __init__(self, path, cfg=Config()):
        super(OptDEMTrajData, self).__init__(path, cfg)

    def __getitem__(self, i, visualize=False):
        cloud = self.get_cloud(i)
        color = self.get_cloud_color(i)

        # merge cloud and colors
        cloud = merge_arrays([cloud, color], flatten=True, usemask=False)

        traj = self.get_traj(i)
        height = self.get_optimized_terrain(i)['height']
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


class MonoDEMData(DEMTrajData):
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

    def get_undistorted_image(self, i, cam):
        img = self.get_image(i, cam)
        for key in self.calib.keys():
            if cam in key:
                cam = key
                break
        K = self.calib[cam]['camera_matrix']['data']
        r, c = self.calib[cam]['camera_matrix']['rows'], self.calib[cam]['camera_matrix']['cols']
        K = np.array(K).reshape((r, c))
        D = self.calib[cam]['distortion_coefficients']['data']
        D = np.array(D)
        img_undist, K = undistort_image(img, K, D)
        return img_undist, K

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

    def calculate_img_statistics(self):
        # calculate mean and std from the entire dataset
        means, stds = [], []
        print('Calculating images mean and std from the entire dataset...')
        for i in tqdm(range(len(self))):
            img = self.get_image(i, camera=self.cameras[0])
            img_01 = normalize(img)

            mean = img_01.reshape([-1, 3]).mean(axis=0)
            std = img_01.reshape([-1, 3]).std(axis=0)

            means.append(mean)
            stds.append(std)

        mean = np.asarray(means).mean(axis=0)
        std = np.asarray(stds).mean(axis=0)

        print(f'Estimated mean: {mean} \n and std: {std}')
        return mean, std

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

        img_front, K = self.get_undistorted_image(i, camera.split('_')[-1])

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
        terrain = self.get_optimized_terrain(i)
        height_opt = terrain['height']

        # crop height map to observation area defined by square grid
        h, w = height_est.shape
        square_grid = square[:, :2] / self.cfg.grid_res + np.asarray([w / 2, h / 2])
        height_est_cam = height_est[int(square_grid[0, 1]):int(square_grid[2, 1]),
                                    int(square_grid[0, 0]):int(square_grid[2, 0])]
        height_opt_cam = height_opt[int(square_grid[0, 1]):int(square_grid[2, 1]),
                                    int(square_grid[0, 0]):int(square_grid[2, 0])]
        # poses in grid coordinates
        poses_grid = poses[:, :2, 3] / self.cfg.grid_res + np.asarray([w / 2, h / 2])
        # crop poses to observation area defined by square grid
        poses_grid_cam = poses_grid[(poses_grid[:, 0] > square_grid[0, 0]) & (poses_grid[:, 0] < square_grid[2, 0]) &
                                    (poses_grid[:, 1] > square_grid[0, 1]) & (poses_grid[:, 1] < square_grid[2, 1])]
        poses_grid_cam -= np.asarray([square_grid[0, 0], square_grid[0, 1]])

        # visited by poses dilated height map area mask
        H, W = height_opt_cam.shape
        kernel = np.ones((3, 3), dtype=np.uint8)
        weights_opt_cam = np.zeros((H, W), dtype=np.uint8)
        poses_grid_cam = poses_grid_cam.astype(np.uint32)
        weights_opt_cam[poses_grid_cam[:, 1], poses_grid_cam[:, 0]] = 1
        weights_opt_cam = cv2.dilate(weights_opt_cam, kernel, iterations=5)
        weights_opt_cam = weights_opt_cam.astype(bool)

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
            height_opt_cam = np.rot90(height_opt_cam, 1)
            weights_opt_cam = np.rot90(weights_opt_cam, 1)
        elif 'right' in camera:
            height_est_cam = np.rot90(height_est_cam, -1)
            height_opt_cam = np.rot90(height_opt_cam, -1)
            weights_opt_cam = np.rot90(weights_opt_cam, -1)
        elif 'rear' in camera:
            height_est_cam = np.rot90(height_est_cam, 2)
            height_opt_cam = np.rot90(height_opt_cam, 2)
            weights_opt_cam = np.rot90(weights_opt_cam, 2)

        # rotate heightmaps to have robot position at the bottom
        height_opt_cam = np.rot90(height_opt_cam, axes=(0, 1))
        height_est_cam = np.rot90(height_est_cam, axes=(0, 1))
        weights_opt_cam = np.rot90(weights_opt_cam, axes=(0, 1))
        weights_est_cam = np.rot90(weights_est_cam, axes=(0, 1))
        # flip heightmaps to have robot position at the bottom
        # we do copy, because of this issue:
        # https://stackoverflow.com/questions/72550211/valueerror-at-least-one-stride-in-the-given-numpy-array-is-negative-and-tensor
        height_opt_cam = np.fliplr(height_opt_cam).copy()
        height_est_cam = np.fliplr(height_est_cam).copy()
        weights_opt_cam = np.fliplr(weights_opt_cam).copy()
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
            plt.imshow(height_opt, cmap='jet', alpha=0.8, origin='lower')
            plt.plot(poses_grid[:, 0], poses_grid[:, 1], 'ro', markersize=2)
            plt.plot(square_grid[:, 0], square_grid[:, 1], 'y--', linewidth=2)
            # plt.colorbar()

            plt.subplot(247)
            plt.title('Optimized heightmap in camera frame')
            plt.imshow(height_opt_cam, cmap='jet', alpha=1.)
            plt.colorbar()

            plt.subplot(248)
            plt.title('Optimized heightmap weights')
            plt.imshow(weights_opt_cam, cmap='gray', alpha=1.)

            # mlab.show()
            plt.show()

        # resize and normalize image
        img_front = self.preprocess_img(img_front)

        # flip image and heightmaps from left to right with 50% probability
        if self.is_train and np.random.random() > 0.5:
            img_front = np.fliplr(img_front).copy()
            height_opt_cam = np.fliplr(height_opt_cam).copy()
            height_est_cam = np.fliplr(height_est_cam).copy()
            weights_opt_cam = np.fliplr(weights_opt_cam).copy()
            weights_est_cam = np.fliplr(weights_est_cam).copy()

        # convert to CHW format
        img_front_CHW = img_front.transpose((2, 0, 1))

        return img_front_CHW, height_opt_cam[None], height_est_cam[None], weights_opt_cam[None], weights_est_cam[None]


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

    def get_image_data(self, i):
        imgs = []
        rots = []
        trans = []
        post_rots = []
        post_trans = []
        intrins = []

        # permute cameras
        cameras = self.cameras.copy()
        if self.is_train:
            np.random.shuffle(cameras)

        for cam in cameras:
            img, K = self.get_undistorted_image(i, cam)

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
            img = standardize_img(np.asarray(img))
            img = torch.as_tensor(img).permute((2, 0, 1))
            K = torch.as_tensor(K)

            # extrinsics
            T_lidar_cam = self.calib['transformations']['T_os_sensor__%s' % cam]['data']
            T_lidar_cam = np.asarray(T_lidar_cam, dtype=float).reshape((4, 4))
            T_cam_lidar = np.linalg.inv(T_lidar_cam)
            T_robot_lidar = self.calib['transformations']['T_base_link__os_sensor']['data']
            T_robot_lidar = np.asarray(T_robot_lidar, dtype=float).reshape((4, 4))
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

    def get_height_map_data(self, i, cached=True):
        cloud = self.get_cloud(i)
        points = position(cloud)

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
            robot_z = np.asarray(self.calib['transformations']['T_base_link__base_footprint']['data'],
                                 dtype=float).reshape(4, 4)[2, 3]
            xyz_mask = estimate_heightmap(points,
                                          d_min=self.cfg.d_min, d_max=self.cfg.d_max,
                                          grid_res=self.cfg.grid_res, h_max=self.cfg.h_max,
                                          hm_interp_method=self.hm_interp_method,
                                          robot_z=robot_z)
            # save height map as numpy array
            result = np.zeros((xyz_mask['z'].shape[0], xyz_mask['z'].shape[1]),
                              dtype=[(key, np.float32) for key in xyz_mask.keys()])
            for key in xyz_mask.keys():
                result[key] = xyz_mask[key]
            os.makedirs(dir_path, exist_ok=True)
            np.save(os.path.join(dir_path, '%s.npy' % self.ids[i]), result)

        heightmap = torch.stack([torch.as_tensor(xyz_mask[i], dtype=torch.float32) for i in ['z', 'mask']])
        # heightmap = torch.as_tensor(xyz_mask['z'])[None]

        return heightmap

    def __getitem__(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(i)
        height = self.get_height_map_data(i)
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
        height = self.get_height_map_data(i)
        lidar_pts = torch.as_tensor(position(self.get_cloud(i))).T
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_pts, height


class OmniOptDEMData(OmniDEMData):
    def __init__(self,
                 path,
                 data_aug_conf,
                 is_train=True,
                 cfg=Config()
                 ):
        super(OmniOptDEMData, self).__init__(path, data_aug_conf, is_train=is_train, cfg=cfg)

    def get_height_map_data(self, i, h_diff=0.1):
        terrain = self.get_optimized_terrain(i)
        height = torch.as_tensor(terrain['height'])
        height0 = torch.as_tensor(terrain['height_init'])
        H, W = height.shape
        h, w = 2 * self.cfg.d_max // self.cfg.grid_res, 2 * self.cfg.d_max // self.cfg.grid_res
        # select only the h x w area from the center of the height map
        height = height[int(H // 2 - h // 2):int(H // 2 + h // 2),
                        int(W // 2 - w // 2):int(W // 2 + w // 2)]
        height0 = height0[int(H // 2 - h // 2):int(H // 2 + h // 2),
                          int(W // 2 - w // 2):int(W // 2 + w // 2)]
        # mask = torch.ones_like(height)
        mask = (height - height0).abs() > h_diff
        height = torch.stack([height, mask])
        return height


class OmniOptDEMDataVis(OmniOptDEMData):
    def __init__(self,
                 path,
                 data_aug_conf,
                 is_train=True,
                 cfg=Config()
                 ):
        super(OmniOptDEMDataVis, self).__init__(path, data_aug_conf, is_train=is_train, cfg=cfg)

    def __getitem__(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(i)
        height = self.get_height_map_data(i)
        lidar_pts = torch.as_tensor(position(self.get_cloud(i))).T
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_pts, height


def segm_demo():
    path = seq_paths[0]
    assert os.path.exists(path)
    ds = SegmentationDataset(path)

    # visualize a sample from the data set
    for i in np.random.choice(range(len(ds)), 1):
        _ = ds.__getitem__(i, visualize=True)


def heightmap_demo():
    from ..vis import show_cloud_plt
    from ..cloudproc import filter_grid, filter_range
    import matplotlib.pyplot as plt

    path = seq_paths[0]
    assert os.path.exists(path)

    cfg = Config()
    # ds = OptDEMTrajData(path, cfg=cfg)
    ds = DEMTrajData(path, cfg=cfg)

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
        ds = DEMTrajData(path, cfg=cfg)

        robot_pose = np.eye(4)
        robot_frame = 'base_link'
        lidar_frame = 'os_sensor'

        Tr_robot_lidar = ds.calib['transformations'][f'T_{robot_frame}__{lidar_frame}']['data']
        Tr_robot_lidar = np.asarray(Tr_robot_lidar, dtype=float).reshape((4, 4))
        camera_frames = ['camera_left', 'camera_right', 'camera_up', 'camera_fisheye_front', 'camera_fisheye_rear'] \
        if 'marv' in path else ['camera_left', 'camera_right', 'camera_front', 'camera_rear']

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
        ds = DEMTrajData(path, cfg=cfg)

        i = np.random.choice(range(len(ds)))
        # i = 10
        cloud = ds.get_cloud(i)
        colors = ds.get_cloud_color(i)

        # poses
        traj = ds.get_traj(i)
        poses = traj['poses']

        # images
        img_front = ds.get_image(i, 'front')
        img_rear = ds.get_image(i, 'rear')
        img_left = ds.get_image(i, 'left')
        img_right = ds.get_image(i, 'right')

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
    from mayavi import mlab
    from ..vis import draw_coord_frames

    path = np.random.choice(seq_paths)
    assert os.path.exists(path)

    cfg = Config()
    cfg.from_yaml(os.path.join(path, 'terrain', 'train_log', 'cfg.yaml'))
    # cfg.d_min = 1.

    ds = DEMTrajData(path, cfg=cfg)
    i = np.random.choice(range(len(ds)))

    # trajectory poses
    poses = ds.get_traj(i)['poses']
    # point cloud
    cloud = ds.get_cloud(i)
    points = position(cloud)

    img = ds.get_image(i)

    # height map: estimated from point cloud
    heightmap = ds.estimate_heightmap(points)
    height_est = heightmap['z']
    x_grid, y_grid = heightmap['x'], heightmap['y']
    # height map: optimized from robot-terrain interaction model
    terrain = ds.get_optimized_terrain(i)
    height = terrain['height']

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

    # # draw height map as a surface
    # mlab.figure(size=(800, 800))
    # # mlab.mesh(x_grid, y_grid, height_est, color=(0, 0, 1), opacity=0.3)
    # mlab.mesh(x_grid, y_grid, height, color=(0, 1, 0), opacity=0.4)
    # # add wireframe
    # mlab.mesh(x_grid, y_grid, height, color=(0, 0, 0), representation='wireframe', opacity=0.2)
    # # draw trajectory
    # mlab.plot3d(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], color=(1, 0, 0), tube_radius=0.02)
    # draw_coord_frames(poses, scale=0.1)

    # # draw point cloud with colors denoting height (from blue to red)
    # # https://stackoverflow.com/questions/54263312/plotting-3d-points-with-different-colors-in-mayavi-python
    # s = points[:, 2]
    # s = (s - s.min()) / (s.max() - s.min())
    # # Create and populate lookup table (the integer index in s corresponding
    # #   to the point will be used as the row in the lookup table
    # lut = np.zeros((len(s), 4))
    # # A simple lookup table that transitions from red (at index 0) to
    # #   blue (at index len(data)-1)
    # for row, f in enumerate(s):
    #     lut[row, :] = [255 * (1 - f), 0, 255 * f, 255]
    #
    # p3d = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], s, scale_mode='none', scale_factor=0.01)
    # p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(s)
    # p3d.module_manager.scalar_lut_manager.lut.table = lut
    # mlab.show()


def vis_train_sample():
    cfg = Config()
    path = np.random.choice(seq_paths)
    cfg.from_yaml(os.path.join(path, 'terrain', 'train_log', 'cfg.yaml'))

    ds = MonoDEMData(path=path, cfg=cfg)
    i = np.random.choice(range(len(ds)))
    # i = 0
    print(f'Visualizing sample {i}...')
    img, height_opt, height_est, weights_opt, weights_est = ds.__getitem__(i, visualize=True)
    # img, height_opt, height_est, weights_opt = ds[i]
    img = img.transpose(1, 2, 0)

    plt.figure(figsize=(20, 7))
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.title('Height Label')
    plt.imshow(height_opt.squeeze(), cmap='jet')
    plt.imshow(weights_opt.squeeze(), alpha=0.5, cmap='gray')
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

    # path = 'robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'
    # # path = 'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-15-01-44_trav/'
    # path = os.path.join(os.path.join(data_dir, path))
    # cfg.from_yaml(os.path.join(path, 'terrain', 'train_log', 'cfg.yaml'))
    #
    # ds = MonoDemDataset(path=path,
    #                     cfg=cfg)
    # i = np.random.choice(range(len(ds)))
    # # i = 0
    # print(f'Visualizing sample {i}...')
    # ds.__getitem__(i, visualize=True)


def vis_estimated_height_map():
    from time import time

    cfg = Config()
    cfg.grid_res = 0.1
    cfg.d_max = 12.8
    cfg.d_min = 1.
    cfg.h_max = 1.
    # cfg.hm_interp_method = None
    cfg.hm_interp_method = 'nearest'

    path = np.random.choice(seq_paths)
    ds = DEMTrajData(path=path, cfg=cfg)

    # # check performance
    # for interp_method in ['nearest', 'linear', 'cubic', None]:
    #     print(f'Interpolation method: {interp_method}')
    #     t = time()
    #     for i in tqdm(range(len(ds))):
    #         cloud = ds.get_cloud(i)
    #         points = position(cloud)
    #         ds.cfg.hm_interp_method = interp_method
    #         ds.estimate_heightmap(points)
    #     print(f'Average time per sample: {(time() - t) / len(ds)} s')

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
    img_raw, _ = ds.get_undistorted_image(i, 'front')
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
    img, height_opt, height_est, weights_opt, weights_est = ds[i]
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
        ds = DEMTrajData(path=path)
        ds.global_cloud(vis=True, step_size=100)


def explore_data(path, grid_conf, data_aug_conf, cfg, modelf=None,
                 sample_range='random', save=False, opt_terrain=False, is_train=False):
    assert os.path.exists(path)
    assert sample_range in ['random', 'all']

    Data = OmniOptDEMDataVis if opt_terrain else OmniDEMDataVis

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    if modelf is not None:
        model.load_state_dict(torch.load(modelf))
        print('Loaded LSS model from', modelf)
        model.eval()

    ds = Data(path, is_train=is_train, data_aug_conf=data_aug_conf, cfg=cfg)

    H, W = data_aug_conf['H'], data_aug_conf['W']
    cams = data_aug_conf['cams']
    rat = H / W
    val = 10.1

    if sample_range == 'random':
        sample_range = [np.random.choice(range(len(ds)))]
    else:
        sample_range = tqdm(range(len(ds)), total=len(ds))

    for sample_i in sample_range:
        fig = plt.figure(figsize=(val + val / 2 * 2 * rat * 2, val / 2 * 2 * rat))
        gs = mpl.gridspec.GridSpec(2, 4, width_ratios=(1, 1, 2 * rat, 2 * rat))
        gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

        sample = ds[sample_i]
        sample = [s[np.newaxis] for s in sample]
        imgs, rots, trans, intrins, post_rots, post_trans, pts, bev_map = sample
        if modelf is not None:
            with torch.no_grad():
                inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
                inputs = [torch.as_tensor(i, dtype=torch.float32) for i in inputs]
                bev_map = model(*inputs)

        img_pts = model.get_geometry(rots, trans, intrins, post_rots, post_trans)

        for si in range(imgs.shape[0]):
            plt.clf()
            final_ax = plt.subplot(gs[:, 3:4])
            for imgi, img in enumerate(imgs[si]):
                ego_pts = ego_to_cam(pts[si], rots[si, imgi], trans[si, imgi], intrins[si, imgi])
                mask = get_only_in_img_mask(ego_pts, H, W)
                plot_pts = post_rots[si, imgi].matmul(ego_pts) + post_trans[si, imgi].unsqueeze(1)

                ax = plt.subplot(gs[imgi // 2, imgi % 2])
                showimg = destandardize_img(img.permute(1, 2, 0))

                plt.imshow(showimg)
                plt.scatter(plot_pts[0, mask], plot_pts[1, mask], c=ego_pts[2, mask], s=2, alpha=0.2, cmap='jet')
                plt.axis('off')
                # camera name as text on image
                plt.text(0.5, 0.9, cams[imgi].replace('_', ' '), horizontalalignment='center', verticalalignment='top',
                         transform=ax.transAxes, fontsize=10)

                plt.sca(final_ax)
                plt.plot(img_pts[si, imgi, :, :, :, 0].view(-1), img_pts[si, imgi, :, :, :, 1].view(-1), '.',
                         label=cams[imgi].replace('_', ' '))

            plt.legend(loc='upper right')
            final_ax.set_aspect('equal')
            plt.xlim((-cfg.d_max, cfg.d_max))
            plt.ylim((-cfg.d_max, cfg.d_max))

            # ax = plt.subplot(gs[:, 2:3])
            # plt.scatter(pts[si, 0], pts[si, 1], c=pts[si, 2], vmin=-0.5, vmax=0.5, s=2, cmap='Greys')
            # plt.xlim((-cfg.d_max, cfg.d_max))
            # plt.ylim((-cfg.d_max, cfg.d_max))
            # ax.set_aspect('equal')

            ax = plt.subplot(gs[:, 2:3])
            plt.imshow(bev_map[si][0], origin='lower', cmap='jet', vmin=-0.5, vmax=0.5)
            # plt.imshow(bev_map[si][1], origin='lower', cmap='Greys', vmin=0., vmax=1.)
            plt.colorbar()

            if save:
                save_dir = os.path.join(path, 'terrain', 'visuals')
                os.makedirs(save_dir, exist_ok=True)
                imname = f'{ds.ids[sample_i]}.jpg'
                imname = os.path.join(save_dir, imname)
                # print('saving', imname)
                plt.savefig(imname)
                plt.close(fig)
            else:
                plt.show()


def main():
    segm_demo()
    heightmap_demo()
    extrinsics_demo()
    vis_rgb_cloud()
    traversed_height_map()
    vis_train_sample()
    vis_hm_weights()
    vis_estimated_height_map()
    vis_img_augs()
    global_cloud_demo()


if __name__ == '__main__':
    main()
