import copy
import os
import numpy as np
import torch
import torchvision
from skimage.draw import polygon
from torch.utils.data import Dataset
from ..models.terrain_encoder.utils import img_transform, normalize_img, resize_img
from ..models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, sample_augmentation
from ..dphys_config import DPhysConfig
from ..transformations import transform_cloud
from ..cloudproc import estimate_heightmap, hm_to_cloud
from ..utils import position, timing, read_yaml
from ..cloudproc import filter_grid
from ..utils import normalize, load_calib
from .coco import COCO_CATEGORIES, COCO_CLASSES
import albumentations as A
from PIL import Image
from tqdm import tqdm
import open3d as o3d


__all__ = [
    'data_dir',
    'ROUGH',
    'rough_seq_paths',
]

monoforce_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
data_dir = os.path.realpath(os.path.join(monoforce_dir, 'data'))

rough_seq_paths = [
        # MARV robot
        os.path.join(data_dir, 'ROUGH/marv/24-08-14-monoforce-long_drive'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-09-26-13-46-51'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-09-26-13-54-43'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-10-05-12-34-53'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-10-05-13-01-40'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-10-05-13-17-08'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-10-05-13-29-39'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-10-05-13-43-21'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-10-05-13-57-57'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-10-05-14-12-29'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-10-05-14-22-10'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-10-05-14-28-15'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-10-31-15-16-42'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-10-31-15-26-47'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-10-31-15-35-05'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-10-31-15-52-07'),
        os.path.join(data_dir, 'ROUGH/marv/marv_2024-10-31-15-56-33'),

        # TRADR robot
        os.path.join(data_dir, 'ROUGH/tradr2/ugv_2024-09-10-17-02-31'),
        os.path.join(data_dir, 'ROUGH/tradr2/ugv_2024-09-10-17-12-12'),
        os.path.join(data_dir, 'ROUGH/tradr2/ugv_2024-09-26-13-54-18'),
        os.path.join(data_dir, 'ROUGH/tradr2/ugv_2024-09-26-13-58-46'),
        os.path.join(data_dir, 'ROUGH/tradr2/ugv_2024-09-26-14-03-57'),
        os.path.join(data_dir, 'ROUGH/tradr2/ugv_2024-09-26-14-14-42'),
        os.path.join(data_dir, 'ROUGH/tradr2/ugv_2024-10-05-15-40-41'),
        os.path.join(data_dir, 'ROUGH/tradr2/ugv_2024-10-05-15-48-31'),
        os.path.join(data_dir, 'ROUGH/tradr2/ugv_2024-10-05-15-58-52'),
        os.path.join(data_dir, 'ROUGH/tradr2/ugv_2024-10-05-16-08-30'),
        os.path.join(data_dir, 'ROUGH/tradr2/ugv_2024-10-05-16-24-48'),
]


class ROUGH(Dataset):
    """
    A dataset for traversability estimation from camera and lidar data.
    """

    def __init__(self, path,
                 lss_cfg=None,
                 dphys_cfg=DPhysConfig(),
                 is_train=False,
                 only_front_cam=False):
        super(Dataset, self).__init__()
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))
        self.cloud_path = os.path.join(path, 'clouds')
        self.traj_path = os.path.join(path, 'trajectories')
        self.poses_path = os.path.join(path, 'poses', 'lidar_poses.csv')
        self.calib_path = os.path.join(path, 'calibration')
        self.controls_path = os.path.join(path, 'controls', 'cmd_vel.csv')
        self.dphys_cfg = dphys_cfg
        self.calib = load_calib(calib_path=self.calib_path)
        self.ids = self.get_ids()
        self.ts, self.poses = self.get_poses(return_stamps=True)
        self.camera_names = self.get_camera_names()

        self.is_train = is_train
        self.only_front_cam = only_front_cam
        self.camera_names = self.camera_names[:1] if only_front_cam else self.camera_names

        # initialize image augmentations
        if lss_cfg is None:
            lss_cfg = read_yaml(os.path.join(monoforce_dir, 'config', 'lss_cfg.yaml'))
        self.lss_cfg = lss_cfg
        self.img_augs = self.get_img_augs()

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

    def get_ids(self):
        ids = [f[:-4] for f in os.listdir(self.cloud_path)]
        ids = sorted(ids)
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
        all_stamps, all_controls = data[:, 0], data[:, 1:]
        all_stamps -= all_stamps[0]  # start time from 0
        time_left = copy.copy(self.ts[i])
        T_horizon, dt = self.dphys_cfg.traj_sim_time, self.dphys_cfg.dt
        time_right = time_left + T_horizon

        # check if the trajectory is out of the control time stamps
        if time_left > all_stamps[-1] or time_right < all_stamps[0]:
            times_horizon = np.arange(0.0, T_horizon, dt)
            controls = np.zeros((len(times_horizon), all_controls.shape[1]))
            return times_horizon, controls

        # find the closest index to the left and right in all times
        il = np.argmin(np.abs(np.asarray(all_stamps) - time_left))
        ir = np.argmin(np.abs(np.asarray(all_stamps) - time_right))
        ir = max(il + 1, ir)
        ir = np.clip(ir, 0, len(all_controls) - 1)
        timestamps = np.asarray(all_stamps[il:ir])
        timestamps = timestamps - timestamps[0]
        controls = all_controls[il:ir]

        times_horizon = np.arange(0.0, T_horizon, dt)
        controls_horizon = np.zeros((len(times_horizon), controls.shape[1]))
        # interpolate controls to the trajectory time stamps
        for j in range(controls.shape[1]):
            controls_horizon[:, j] = np.interp(times_horizon, timestamps, controls[:, j], left=0.0, right=0.0)

        assert len(times_horizon) == len(controls_horizon), f'Velocity and time stamps have different lengths'
        assert len(times_horizon) == int(T_horizon / dt), f'Velocity and time stamps have different lengths'
        times_horizon = torch.as_tensor(times_horizon, dtype=torch.float32)
        controls_horizon = torch.as_tensor(controls_horizon, dtype=torch.float32)

        return times_horizon, controls_horizon

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
        # TODO: measure velocities and angular velocities with IMU
        # estimating velocities and angular velocities from the trajectory positions for now
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

        states = [xs.reshape([n_states, 3]),
                  xds.reshape([n_states, 3]),
                  Rs.reshape([n_states, 3, 3]),
                  omegas.reshape([n_states, 3])]

        # to torch tensors
        ts = torch.as_tensor(ts)
        states = [torch.as_tensor(s) for s in states]

        return ts, states

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
        # move points to robot frame
        Tr = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr = np.asarray(Tr, dtype=float).reshape((4, 4))
        cloud = transform_cloud(cloud, Tr)
        return cloud

    def get_geom_height_map(self, i, cached=True, dir_name=None, **kwargs):
        """
        Get height map from lidar point cloud.
        :param i: index of the sample
        :param cached: if True, load height map from file if it exists, otherwise estimate it
        :param dir_name: directory to save/load height map
        :param kwargs: additional arguments for height map estimation
        :return: height map (2 x H x W), where 2 is the number of channels (z and mask)
        """
        if dir_name is None:
            dir_name = os.path.join(self.path, 'terrain', 'geom')
        file_path = os.path.join(dir_name, f'{self.ids[i]}.npy')
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

    def get_img_augs(self):
        return A.Compose([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, always_apply=False, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.MotionBlur(blur_limit=7, p=0.5),
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200),
                             p=0.5),
                A.RandomSunFlare(src_radius=100, num_flare_circles_lower=1, num_flare_circles_upper=2, p=0.5),
                A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=0.5),
        ])

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

    def get_image(self, i, camera=None):
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
        return img, K

    def get_cached_resized_img(self, i, camera=None):
        cache_dir = os.path.join(self.path, 'images', 'resized')
        os.makedirs(cache_dir, exist_ok=True)
        cached_img_path = os.path.join(cache_dir, '%s_%s.png' % (self.ids[i], camera))
        if os.path.exists(cached_img_path):
            img = Image.open(cached_img_path)
            assert img is not None, f'Image path {cached_img_path} does not exist'
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
            img, K = self.get_cached_resized_img(i, cam)

            # apply additional image augmentations like blur, brightness, contrast, fog, rain, snow, sun flare
            if self.is_train:
                img = np.array(img)
                img = self.img_augs(image=img)['image']
                img = Image.fromarray(img)

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
    
    def get_semantic_cloud(self, i, classes=None, vis=False):
        if classes is None:
            classes = np.copy(COCO_CLASSES)
        # ids of classes in COCO
        selected_labels = []
        for c in classes:
            if c in COCO_CLASSES:
                selected_labels.append(COCO_CLASSES.index(c))

        lidar_points = position(self.get_cloud(i))
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

    def get_terrain_height_map(self, i, cached=True, dir_name=None):
        """
        Get height map from trajectory points.
        :param i: index of the sample
        :param cached: if True, load height map from file if it exists, otherwise estimate it
        :param dir_name: directory to save/load height map
        :param rigid_classes: classes of obstacles to include in the height map
        :return: heightmap (2 x H x W), where 2 is the number of channels (z and mask)
        """
        if dir_name is None:
            dir_name = os.path.join(self.path, 'terrain', 'rigid')

        file_path = os.path.join(dir_name, f'{self.ids[i]}.npy')
        if cached and os.path.exists(file_path):
            hm_rigid = np.load(file_path, allow_pickle=True).item()
        else:
            traj_points = self.get_footprint_traj_points(i)
            soft_classes = self.lss_cfg['soft_classes']
            rigid_classes = [c for c in COCO_CLASSES if c not in soft_classes]
            seg_points, _ = self.get_semantic_cloud(i, classes=rigid_classes, vis=False)
            points = np.concatenate((seg_points, traj_points), axis=0)
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
        hm_terrain = self.get_terrain_height_map(i)
        if self.only_front_cam:
            mask = self.front_height_map_mask()
            hm_terrain[1] = hm_terrain[1] * torch.from_numpy(mask)
        return (imgs, rots, trans, intrins, post_rots, post_trans,
                hm_terrain,
                control_ts, controls,
                traj_ts, Xs, Xds, Rs, Omegas)
