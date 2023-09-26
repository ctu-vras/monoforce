import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from torch.utils.data import Dataset
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured, merge_arrays
from matplotlib import cm
from mayavi import mlab
from ..config import Config
from ..segmentation import color, position
from ..transformations import transform_cloud
from ..segmentation import position
from ..segmentation import filter_grid, filter_range
from ..imgproc import undistort_image, project_cloud_to_image
from tqdm import tqdm
from .augmentations import horizontal_shift
from ..vis import show_cloud, draw_coord_frame, draw_coord_frames, set_axes_equal
from ..utils import normalize
import yaml
import cv2
import albumentations as A


__all__ = [
    'SegmentationDataset',
    'RobinGasDataset',
    'MonoDemDataset',
    'seq_paths',
]

IGNORE_LABEL = 255
data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'))

seq_paths = [
        os.path.join(data_dir, 'robingas/data/22-10-27-unhost-final-demo/husky_2022-10-27-15-33-57_trav/'),
        os.path.join(data_dir, 'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-15-01-44_trav/'),
        os.path.join(data_dir, 'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-10-33-15_trav/'),
        os.path.join(data_dir, 'robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-16-37-03_trav/'),
        os.path.join(data_dir, 'robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'),
]


class SegmentationDataset(Dataset):
    """
    Class to wrap semi-supervised traversability data generated using lidar odometry and IMU.
    Please, have a look at the `save_clouds_and_trajectories_from_bag` script for data generation from bag file.
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

        points = structured_to_unstructured(cloud[['x', 'y', 'z']])
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


class RobinGasDataset(Dataset):
    """
    Class to wrap semi-supervised traversability data generated using lidar odometry and IMU.
    Please, have a look at the `save_clouds_and_trajectories_from_bag` script for data generation from bag file.
    The dataset additionally contains camera images, calibration data,
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
    """

    def __init__(self, path, cfg=Config()):
        super(Dataset, self).__init__()
        self.path = path
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
        self.calib = self.get_calibrations()
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
        # transform to base footprint frame
        T = self.calib['transformations']['T_base_link__base_footprint']
        T = np.asarray(T['data'], dtype=float).reshape((T['rows'], T['cols']))
        traj['poses'] = np.asarray([T @ pose for pose in traj['poses']])

        return traj

    def get_calibrations(self):
        calib = {}
        # read camera calibration
        cams_path = os.path.join(self.calib_path, 'cameras')
        for file in os.listdir(cams_path):
            if file.endswith('.yaml'):
                with open(os.path.join(cams_path, file), 'r') as f:
                    cam_info = yaml.load(f, Loader=yaml.FullLoader)
                    calib[file.replace('.yaml', '')] = cam_info
                f.close()
        # read cameras-lidar transformations
        trans_path = os.path.join(self.calib_path, 'transformations.yaml')
        with open(trans_path, 'r') as f:
            transforms = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
        calib['transformations'] = transforms
        return calib

    def get_cloud(self, i):
        ind = self.ids[i]
        cloud = np.load(os.path.join(self.cloud_path, '%s.npz' % ind))['cloud']
        if cloud.ndim == 2:
            cloud = cloud.reshape((-1,))
        return cloud

    def get_cloud_color(self, i):
        ind = self.ids[i]
        rgb = np.load(os.path.join(self.cloud_color_path, '%s.npz' % ind))['rgb']
        # convert to structured numpy array with 'r', 'g', 'b' fields
        color = unstructured_to_structured(rgb, names=['r', 'g', 'b'])
        return color

    def get_image(self, i, camera='front'):
        assert camera in ['front', 'rear', 'left', 'right', 'up'], 'Unknown camera: %s' % camera
        prefix = 'camera_fisheye_' if 'marv' in self.path and camera in ['front', 'rear'] else 'camera_'
        camera_name = prefix + camera
        ind = self.ids[i]
        image = cv2.imread(os.path.join(self.path, 'images', '%s_%s.png' % (ind, camera_name)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_optimized_terrain(self, i):
        ind = self.ids[i]
        terrain = np.load(os.path.join(self.path, 'terrain', '%s.npy' % ind))
        # TODO: transformation issue (need to add a transposition hack)
        return terrain.T

    def estimate_heightmap(self, points, fill_value=None, return_filtered_points=False):
        assert points.ndim == 2
        assert points.shape[1] >= 3  # (N x 3)
        assert self.cfg.hm_interp_method in ['linear', 'nearest', 'cubic', None]

        # filter height outliers points
        z = points[:, 2]
        h_min = z[z > np.percentile(z, 2)].min()
        h_max = z[z < np.percentile(z, 98)].max()
        points = points[points[:, 2] > h_min]
        points = points[points[:, 2] < h_max]

        # height above ground
        points = points[points[:, 2] < self.cfg.h_max]

        # filter point cloud in a square
        mask_x = np.logical_and(points[:, 0] >= -self.cfg.d_max, points[:, 0] <= self.cfg.d_max)
        mask_y = np.logical_and(points[:, 1] >= -self.cfg.d_max, points[:, 1] <= self.cfg.d_max)
        mask = np.logical_and(mask_x, mask_y)
        points = points[mask]

        if fill_value is None:
            fill_value = points[:, 2].min()

        # robot points
        robot_mask = filter_range(points, min=0., max=self.cfg.d_min if self.cfg.d_min > 0. else 0., return_mask=True)[1]
        # points = points[~robot_mask]
        points[robot_mask] = np.asarray([0., 0., fill_value])

        # create a grid
        n = int(2 * self.cfg.d_max / self.cfg.grid_res)
        xi = np.linspace(-self.cfg.d_max, self.cfg.d_max, n)
        yi = np.linspace(-self.cfg.d_max, self.cfg.d_max, n)
        x_grid, y_grid = np.meshgrid(xi, yi)

        if self.cfg.hm_interp_method is None:
            # estimate heightmap
            z_grid = np.full(x_grid.shape, fill_value=fill_value)
            for i in range(len(points)):
                x = points[i, 0]
                y = points[i, 1]
                z = points[i, 2]
                # find the closest grid point
                idx_x = np.argmin(np.abs(x_grid[0, :] - x))
                idx_y = np.argmin(np.abs(y_grid[:, 0] - y))
                # update heightmap
                if z > z_grid[idx_y, idx_x] or z_grid[idx_y, idx_x] == fill_value:
                    z_grid[idx_y, idx_x] = z
                else:
                    # print('Point is lower than the current heightmap value, skipping...')
                    pass
        else:
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            z_grid = griddata((x, y), z, (xi[None, :], yi[:, None]),
                              method=self.cfg.hm_interp_method, fill_value=fill_value)

        heightmap = {'x': np.asarray(x_grid, dtype=float),
                     'y': np.asarray(y_grid, dtype=float),
                     'z': np.asarray(z_grid, dtype=float),
                     'mask': z_grid != fill_value}

        if return_filtered_points:
            return heightmap, points

        return heightmap

    def global_cloud(self, colorize=False, vis=False, step_size=1):
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
        poses = traj['poses']

        # move poses, points to robot frame
        Tr = np.linalg.inv(poses[0])
        cloud = transform_cloud(cloud, Tr)
        points = transform_cloud(points, Tr)
        poses = np.asarray([Tr @ pose for pose in poses])
        traj['poses'] = poses

        h_min = poses[:, 2, 3].min()
        height = self.estimate_heightmap(points, fill_value=h_min)

        return cloud, traj, height

    def __len__(self):
        return len(self.ids)


class MonoDemDataset(RobinGasDataset):
    def __init__(self,
                 path,
                 img_size=(512, 512),
                 cameras=None,
                 is_train=False,
                 random_camera_selection_prob=0.2,
                 cfg=Config()):
        super(MonoDemDataset, self).__init__(path, cfg)
        self.img_size = img_size
        self.random_camera_selection_prob = random_camera_selection_prob

        img_statistics_path = os.path.join(self.path, 'calibration', 'img_statistics.yaml')
        if not os.path.exists(img_statistics_path):
            self.img_mean, self.img_std = self.calculate_img_statistics()
            # save to yaml file
            with open(img_statistics_path, 'w') as f:
                yaml.dump({'mean': self.img_mean.tolist(), 'std': self.img_std.tolist()}, f)
        else:
            # load from yaml file
            with open(img_statistics_path, 'r') as f:
                img_statistics = yaml.load(f, Loader=yaml.FullLoader)
                img_mean = img_statistics['mean']
                img_std = img_statistics['std']
            self.img_mean = np.asarray(img_mean)
            self.img_std = np.asarray(img_std)

        self.cameras = ['camera_fisheye_front' if 'marv' in self.path else 'camera_front',
                        'camera_fisheye_rear' if 'marv' in self.path else 'camera_rear',
                        'camera_right',
                        'camera_left'] if cameras is None else cameras

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
        ]) if is_train else None
        self.is_train = is_train

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

    def standardize_img(self, img):
        H, W, C = img.shape
        img_01 = normalize(img)
        img_norm = (img_01 - self.img_mean.reshape((1, 1, C))) / self.img_std.reshape((1, 1, C))
        return img_norm

    def destandardize_img(self, img_norm):
        H, W, C = img_norm.shape
        img_01 = img_norm * self.img_std.reshape((1, 1, C)) + self.img_mean.reshape((1, 1, C))
        return img_01

    def preprocess_img(self, img_raw):
        img = self.resize_crop_img(img_raw)
        if self.is_train:
            img = self.img_augs(image=img)['image']
        # img = self.standardize_img(img)
        return img

    def calculate_img_statistics(self):
        # calculate mean and std from the entire dataset
        means, stds = [], []
        print('Calculating images mean and std from the entire dataset...')
        for i in tqdm(range(len(self))):
            img = self.get_image(i)
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

        cloud = self.get_cloud(i)
        traj = self.get_traj(i)

        points = position(cloud)
        poses = traj['poses']

        img_front, K = self.get_undistorted_image(i, camera.split('_')[-1])

        if visualize:
            # find transformation between camera and lidar
            lidar_to_camera = self.calib['transformations']['T_%s__%s' % (lidar, camera)]['data']
            lidar_to_camera = np.asarray(lidar_to_camera).reshape((4, 4))

            # transform point points to camera frame
            points_cam = transform_cloud(points, lidar_to_camera)

            # project points to image
            points_fov, colors_view, fov_mask = project_cloud_to_image(points_cam, img_front, K, return_mask=True, debug=False)

            # set colors from a particular camera viewpoint
            colors = np.zeros_like(points)
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

        # move poses, points to robot frame
        Tr = np.linalg.inv(poses[0])
        points = transform_cloud(points, Tr)
        poses = np.asarray([Tr @ pose for pose in poses])

        # height map from point cloud (!!! assumes points are in robot frame)
        interpolation = self.cfg.hm_interp_method if self.cfg.hm_interp_method is not None else 'no_interp'
        dir_path = os.path.join(self.path, 'terrain', 'estimated', interpolation)
        # if height map was estimated before - load it
        if os.path.exists(os.path.join(dir_path, '%s.npy' % self.ids[i])):
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
        # we do copy, because of this issue:
        # https://stackoverflow.com/questions/72550211/valueerror-at-least-one-stride-in-the-given-numpy-array-is-negative-and-tensor
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


def segm_demo():
    path = '/home/ruslan/data/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'
    # path = '/home/ruslan/data/robingas/data/22-09-27-unhost/husky/husky_2022-09-27-15-01-44_trav/'
    assert os.path.exists(path)
    ds = SegmentationDataset(path)

    # visualize a sample from the data set
    for i in np.random.choice(range(len(ds)), 1):
        _ = ds.__getitem__(i, visualize=True)


def heightmap_demo():
    from ..vis import show_cloud_plt
    from ..segmentation import filter_grid, filter_range
    import matplotlib.pyplot as plt

    path = '/home/ruslan/data/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'
    # path = '/home/ruslan/data/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-16-37-03_trav/'
    # path = '/home/ruslan/data/robingas/data/22-09-27-unhost/husky/husky_2022-09-27-15-01-44_trav/'
    assert os.path.exists(path)

    cfg = Config()
    cfg.d_min = 1.
    cfg.d_max = 12.8
    cfg.grid_res = 0.1
    cfg.h_max = 2.

    ds = RobinGasDataset(path, cfg=cfg)

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


def calibs_demo():
    from mayavi import mlab
    from ..vis import draw_coord_frames, draw_coord_frame

    path = '/home/ruslan/data/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'
    assert os.path.exists(path)

    cfg = Config()
    ds = RobinGasDataset(path, cfg=cfg)

    lidar_pose = np.eye(4)
    lidar_frame = 'os_sensor'
    camera_frames = ['camera_left', 'camera_right', 'camera_up', 'camera_fisheye_front', 'camera_fisheye_rear']

    poses = []
    for frame in camera_frames:
        pose = ds.calib['transformations'][f'T_{lidar_frame}__{frame}']['data']
        pose = np.asarray(pose).reshape((4, 4))
        pose = np.linalg.inv(pose)
        poses.append(pose[np.newaxis])
    poses = np.concatenate(poses, axis=0)

    # draw coordinate frames
    mlab.figure(size=(800, 800))
    draw_coord_frame(lidar_pose, scale=0.5)
    draw_coord_frames(poses, scale=0.1)
    mlab.show()


def colored_clouds_demo():
    from ..vis import show_cloud
    from ..utils import normalize

    path = '/home/ruslan/data/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'
    assert os.path.exists(path)

    cfg = Config()
    ds = RobinGasDataset(path, cfg=cfg)

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


def terrain_demo():
    from mayavi import mlab
    from ..vis import draw_coord_frames

    # path = '/home/ruslan/data/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'
    path = '/home/ruslan/data/robingas/data/22-09-27-unhost/husky/husky_2022-09-27-15-01-44_trav/'
    assert os.path.exists(path)

    cfg = Config()
    cfg.from_yaml(os.path.join(path, 'terrain', 'train_log', 'cfg.yaml'))
    # cfg.d_min = 1.

    ds = RobinGasDataset(path, cfg=cfg)
    i = np.random.choice(range(len(ds)))

    # trajectory poses
    poses = ds.get_traj(i)['poses']
    # point cloud
    cloud = ds.get_cloud(i)
    points = position(cloud)

    img = ds.get_image(i)

    # transform point cloud to robot frame
    Tr = np.linalg.inv(poses[0])
    points = transform_cloud(points, Tr)
    poses = np.asarray([np.dot(Tr, pose) for pose in poses])

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


def monodem_demo():
    cfg = Config()
    path = '/home/ruslan/data/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'
    # path = '/home/ruslan/data/robingas/data/22-09-27-unhost/husky/husky_2022-09-27-15-01-44_trav/'
    cfg.from_yaml(os.path.join(path, 'terrain', 'train_log', 'cfg.yaml'))

    # camera = 'camera_fisheye_front'
    # camera = 'camera_fisheye_rear'
    # camera = 'camera_left'
    camera = 'camera_right'
    # camera = 'camera_front'
    # camera = 'camera_rear'

    ds = MonoDemDataset(path=path,
                        img_size=(512, 512),
                        cfg=cfg)
    i = np.random.choice(range(len(ds)))
    # i = 0
    print(f'Visualizing sample {i}...')
    img, height_opt, height_est, weights_opt, weights_est = ds.__getitem__(i, visualize=True)
    # img, height_opt, height_est, weights_opt = ds[i]
    img = img.transpose(1, 2, 0) * ds.img_std + ds.img_mean

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


def weights_demo():
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

    # path = '/home/ruslan/data/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'
    # # path = '/home/ruslan/data/robingas/data/22-09-27-unhost/husky/husky_2022-09-27-15-01-44_trav/'
    # cfg.from_yaml(os.path.join(path, 'terrain', 'train_log', 'cfg.yaml'))
    #
    # ds = MonoDemDataset(path=path,
    #                     img_size=(512, 512),
    #                     cfg=cfg)
    # i = np.random.choice(range(len(ds)))
    # # i = 0
    # print(f'Visualizing sample {i}...')
    # ds.__getitem__(i, visualize=True)


def estimate_heightmap_from_cloud():
    from time import time

    cfg = Config()
    cfg.grid_res = 0.1
    cfg.d_max = 12.8
    cfg.d_min = 1.
    cfg.h_max = 1.3
    # cfg.hm_interp_method = None
    cfg.hm_interp_method = 'nearest'

    path = '/home/ruslan/data/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'
    ds = RobinGasDataset(path=path, cfg=cfg)

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


def augs_demo():
    data_path = '/home/ruslan/data/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'
    cfg = Config()
    ds = MonoDemDataset(path=data_path,
                        img_size=(512, 512),
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
        ds = RobinGasDataset(path=path)
        ds.global_cloud(vis=True, step_size=100)


def main():
    # segm_demo()
    # heightmap_demo()
    # calibs_demo()
    # colored_clouds_demo()
    # terrain_demo()
    # monodem_demo()
    # weights_demo()
    # estimate_heightmap_from_cloud()
    # augs_demo()
    global_cloud_demo()


if __name__ == '__main__':
    main()
