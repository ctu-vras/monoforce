import copy
import os
import numpy as np
import torch
import torchvision
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from ..models.terrain_encoder.utils import img_transform, normalize_img, resize_img
from ..models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, sample_augmentation
from ..models.traj_predictor.dphys_config import DPhysConfig
from ..transformations import transform_cloud, position
from ..cloudproc import estimate_heightmap, hm_to_cloud
from ..utils import position, read_yaml
from ..cloudproc import filter_grid
from ..utils import normalize, load_calib
from .wildscenes import METAINFO as WILDSCENES_METAINFO
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
        os.path.join(data_dir, 'ROUGH/24-08-14-monoforce-long_drive'),
        os.path.join(data_dir, 'ROUGH/marv_2024-09-26-13-46-51'),
        os.path.join(data_dir, 'ROUGH/marv_2024-09-26-13-54-43'),
        os.path.join(data_dir, 'ROUGH/marv_2024-10-31-15-16-42'),
        os.path.join(data_dir, 'ROUGH/marv_2024-10-31-15-26-47'),
        os.path.join(data_dir, 'ROUGH/marv_2024-10-31-15-35-05'),
        os.path.join(data_dir, 'ROUGH/marv_2024-10-31-15-52-07'),
        os.path.join(data_dir, 'ROUGH/marv_2024-10-31-15-56-33'),
]


class ROUGH(Dataset):
    """
    A dataset for traversability estimation from camera and lidar data.
    """

    def __init__(self, path,
                 lss_cfg=None,
                 dphys_cfg=None,
                 is_train=False):
        super(Dataset, self).__init__()
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))
        self.cloud_path = os.path.join(path, 'clouds')
        self.traj_path = os.path.join(path, 'trajectories')
        self.poses_path = os.path.join(path, 'poses', 'lidar_poses.csv')
        self.calib_path = os.path.join(path, 'calibration')
        self.controls_path = os.path.join(path, 'controls', 'cmd_vel.csv')
        self.dphys_cfg = dphys_cfg if dphys_cfg is not None else DPhysConfig()
        self.calib = load_calib(calib_path=self.calib_path)
        self.ids = self.get_ids()
        self.poses_ts, self.poses = self.get_all_poses(return_stamps=True)
        self.camera_names = self.get_camera_names()

        self.is_train = is_train

        if lss_cfg is None:
            lss_cfg = read_yaml(os.path.join(monoforce_dir, 'config', 'lss_cfg.yaml'))
        self.lss_cfg = lss_cfg
        self.grid_res = lss_cfg['grid_conf']['xbound'][2]

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

    def get_ids(self):
        ids = [f[:-4] for f in os.listdir(self.cloud_path)]
        ids = sorted(ids)
        return ids

    @staticmethod
    def pose2mat(pose):
        T = np.eye(4)
        T[:3, :4] = pose.reshape((3, 4))
        return T

    def get_all_poses(self, return_stamps=False):
        if not os.path.exists(self.poses_path):
            print(f'Poses file {self.poses_path} does not exist')
            return None, None if return_stamps else None
        data = np.loadtxt(self.poses_path, delimiter=',', skiprows=1)
        assert len(data) > 0, f'No poses found in {self.poses_path}'
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

    def ind_to_stamp(self, i):
        ind = self.ids[i]
        stamp = float(ind.replace('_', '.'))
        return stamp

    def get_pose(self, i):
        stamp = self.ind_to_stamp(i)
        pose_i = np.argmin(np.abs(self.poses_ts - stamp))
        pose = self.poses[pose_i]
        return pose

    def get_initial_pose_on_heightmap(self, i):
        map_pose = self.get_pose(i)
        roll, pitch, yaw = Rotation.from_matrix(map_pose[:3, :3]).as_euler('xyz')
        R = Rotation.from_euler('xyz', [roll, pitch, 0]).as_matrix()
        pose_gravity_aligned = np.eye(4)
        pose_gravity_aligned[:3, :3] = R
        return pose_gravity_aligned

    def get_all_controls(self):
        if not os.path.exists(self.controls_path):
            print(f'Controls file {self.controls_path} does not exist')
            return None, None
        data = np.loadtxt(self.controls_path, delimiter=',', skiprows=1)
        assert len(data) > 0, f'No controls found in {self.controls_path}'
        all_control_stamps, all_controls = data[:, 0], data[:, 1:]
        return all_control_stamps, all_controls

    def get_controls(self, i):
        all_control_stamps, all_controls = self.get_all_controls()
        # assert all_controls is not None, f'Controls are not available'
        time_left = self.ind_to_stamp(i)
        # start time from 0
        time_left -= all_control_stamps[0]
        all_control_stamps -= all_control_stamps[0]
        T_horizon, dt = self.dphys_cfg.traj_sim_time, self.dphys_cfg.dt
        time_right = time_left + T_horizon

        # check if the trajectory is out of the control time stamps
        if time_left > all_control_stamps[-1] or time_right < all_control_stamps[0]:
            # print(f'Trajectory is out of the recorded control time stamps. Using zero controls.')
            control_stamps_horizon = torch.arange(0.0, T_horizon, dt, dtype=torch.float32)
            controls = torch.zeros((len(control_stamps_horizon), all_controls.shape[1]), dtype=torch.float32)
            return control_stamps_horizon, controls

        # find the closest index to the left and right in all times
        il = np.argmin(np.abs(np.asarray(all_control_stamps) - time_left))
        ir = np.argmin(np.abs(np.asarray(all_control_stamps) - time_right))
        ir = min(max(il + 1, ir), len(all_control_stamps))
        control_stamps = np.asarray(all_control_stamps[il:ir])
        control_stamps = control_stamps - control_stamps[0]
        controls = all_controls[il:ir]

        control_stamps_horizon = np.arange(0.0, T_horizon, dt)
        controls_horizon = np.zeros((len(control_stamps_horizon), controls.shape[1]))
        # interpolate controls to the trajectory time stamps
        for j in range(controls.shape[1]):
            controls_horizon[:, j] = np.interp(control_stamps_horizon, control_stamps, controls[:, j], left=0.0, right=0.0)

        assert len(control_stamps_horizon) == len(controls_horizon), f'Velocity and time stamps have different lengths'
        assert len(control_stamps_horizon) == int(T_horizon / dt), f'Velocity and time stamps have different lengths'
        control_stamps_horizon = torch.as_tensor(control_stamps_horizon, dtype=torch.float32)
        controls_horizon = torch.as_tensor(controls_horizon, dtype=torch.float32)

        return control_stamps_horizon, controls_horizon

    @staticmethod
    def get_camera_names():
        # cams_yaml = os.listdir(os.path.join(self.path, 'calibration/cameras'))
        # cams = sorted([cam.replace('.yaml', '') for cam in cams_yaml])
        cams = ['camera_left', 'camera_front', 'camera_right', 'camera_rear']
        return cams

    def get_traj(self, i, T_horizon=None):
        # n_frames equals to the number of future poses (trajectory length)
        if T_horizon is None:
            T_horizon = self.dphys_cfg.traj_sim_time
        dt = 0.1  # lidar frequency is 10 Hz

        # get trajectory as sequence of `n_frames` future poses
        all_poses = copy.copy(self.poses)
        all_ts = copy.copy(self.poses_ts)
        time_left = self.ind_to_stamp(i)
        il = np.argmin(np.abs(self.poses_ts - time_left))
        ir = np.argmin(np.abs(all_ts - (self.poses_ts[il] + T_horizon)))
        ir = min(max(ir, il+1), len(all_ts))
        poses = all_poses[il:ir]
        stamps = np.asarray(all_ts[il:ir])

        # transform poses to the same coordinate frame as the height map
        poses = np.linalg.inv(poses[0]) @ poses
        stamps = stamps - stamps[0]

        # limit stamps and poses to the horizon
        stamps = stamps[stamps <= T_horizon]
        poses = poses[:len(stamps)]

        # make sure the trajectory has the fixed length
        n_frames = int(np.ceil(T_horizon / dt))
        if len(poses) < n_frames:
            # repeat the last pose to fill the trajectory
            poses = np.concatenate([poses, np.tile(poses[-1:], (n_frames - len(poses), 1, 1))], axis=0)
            stamps = np.concatenate([stamps, stamps[-1] + np.arange(1, n_frames - len(stamps) + 1) * dt], axis=0)
            assert len(poses) == n_frames, f'Poses and stamps have different lengths {len(poses)} != {n_frames}'
        # truncate the trajectory
        poses = poses[:n_frames]
        stamps = stamps[:n_frames]
        assert len(poses) == len(stamps), f'Poses and time stamps have different lengths'
        assert len(poses) == n_frames

        # gravity-aligned poses
        pose_grav_aligned = self.get_initial_pose_on_heightmap(i)
        pose_grav_aligned = np.asarray(pose_grav_aligned, dtype=poses.dtype)
        poses = pose_grav_aligned @ poses

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
        ts = torch.as_tensor(ts, dtype=torch.float32)
        states = [torch.as_tensor(s, dtype=torch.float32) for s in states]

        return ts, states

    def get_raw_cloud(self, i):
        ind = self.ids[i]
        cloud_path = os.path.join(self.cloud_path, '%s.npz' % ind)
        assert os.path.exists(cloud_path), f'Cloud path {cloud_path} does not exist'
        cloud = np.load(cloud_path)['cloud']
        if cloud.ndim == 2:
            cloud = cloud.reshape((-1,))
        return cloud

    def get_cloud(self, i, gravity_aligned=True):
        cloud = self.get_raw_cloud(i)
        # move points to robot frame
        Tr = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr = np.asarray(Tr, dtype=float).reshape((4, 4))
        cloud = transform_cloud(cloud, Tr)
        if gravity_aligned:
            # gravity-alignment
            pose_gravity_aligned = self.get_initial_pose_on_heightmap(i)
            cloud = transform_cloud(cloud, pose_gravity_aligned)
        return cloud

    def get_geom_height_map(self, i, cached=True, dir_name=None):
        """
        Get height map from lidar point cloud.
        :param i: index of the sample
        :param cached: if True, load height map from file if it exists, otherwise estimate it
        :param dir_name: directory to save/load heightmap
        :return: heightmap (2 x H x W), where 2 is the number of channels (z and mask)
        """
        if dir_name is None:
            dir_name = os.path.join(self.path, 'terrain', 'geom')
        file_path = os.path.join(dir_name, f'{self.ids[i]}.npy')
        if cached and os.path.exists(file_path):
            lidar_hm = np.load(file_path)
        else:
            points = torch.as_tensor(position(self.get_cloud(i)))
            lidar_hm = estimate_heightmap(points, d_max=self.dphys_cfg.d_max,
                                          grid_res=self.grid_res,
                                          h_max=self.dphys_cfg.h_max,
                                          r_min=self.dphys_cfg.r_min)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, lidar_hm.cpu().numpy())
        heightmap = torch.as_tensor(lidar_hm)
        return heightmap

    def get_footprint_traj_points(self, i, robot_size=(0.7, 1.0), T_horizon=None):
        # robot footprint points grid
        width, length = robot_size
        x = np.arange(-length / 2, length / 2, self.grid_res)
        y = np.arange(-width / 2, width / 2, self.grid_res)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        footprint0 = np.stack([x, y, z], axis=-1).reshape((-1, 3))
        footprint0 = np.asarray(footprint0, dtype=np.float32)

        Tr_base_link__base_footprint = np.asarray(self.calib['transformations']['T_base_link__base_footprint']['data'],
                                                  dtype=np.float32).reshape((4, 4))
        traj = self.get_traj(i, T_horizon=T_horizon)
        poses = traj['poses']
        poses_footprint = poses
        poses_footprint[:, 2, 3] -= abs(Tr_base_link__base_footprint[2, 3])  # subtract robot's clearance

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
                cloud = self.get_cloud(i, gravity_aligned=False)
                T = self.get_pose(i)
                cloud = transform_cloud(cloud, T)
                points = position(cloud)
                points = filter_grid(points, self.grid_res, keep='first', log=False)
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

            poses = self.get_all_poses()
            pcd_poses = o3d.geometry.PointCloud()
            pcd_poses.points = o3d.utility.Vector3dVector(poses[:, :3, 3])
            pcd_poses.paint_uniform_color([0.8, 0.1, 0.1])

            # o3d.visualization.draw_geometries([pcd_poses])
            o3d.visualization.draw_geometries([pcd, pcd_poses])
        return global_cloud

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
        if camera is None:
            camera = self.camera_names[0]
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

        pose_grav_aligned = self.get_initial_pose_on_heightmap(i)
        R = pose_grav_aligned[:3, :3]

        for cam in self.camera_names:
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
            # gravity-aligned pose
            T_robot_cam[:3, :3] = R @ T_robot_cam[:3, :3]
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
        label_2_rgb = {cidx: p for cidx, p in zip(WILDSCENES_METAINFO['cidx'], WILDSCENES_METAINFO['palette'])}
        seg_color = np.zeros(list(seg_label.shape) + [3], dtype=np.float32)
        for cidx, c in label_2_rgb.items():
            seg_color[seg_label == cidx] = c
        seg_color /= 255.
        return seg_color

    def get_seg_label(self, i, camera=None):
        if camera is None:
            camera = self.camera_names[0]
        id = self.ids[i]
        seg_path = os.path.join(self.path, 'images/wildscenes_seg/seg/', '%s_%s.png' % (id, camera))
        assert os.path.exists(seg_path), f'Image path {seg_path} does not exist'
        seg = Image.open(seg_path)
        size = self.get_raw_img_size(i, camera)
        transform = torchvision.transforms.Resize(size)
        seg = transform(seg)
        return seg

    def get_seg_vis(self, i, camera=None):
        if camera is None:
            camera = self.camera_names[0]
        id = self.ids[i]
        seg_path = os.path.join(self.path, 'images/wildscenes_seg/vis/', '%s_%s.png' % (id, camera))
        assert os.path.exists(seg_path), f'Image path {seg_path} does not exist'
        seg = Image.open(seg_path)
        return seg

    def get_semantic_cloud(self, i, classes=None, vis=False):
        mi = WILDSCENES_METAINFO
        if classes is None:
            classes = mi['classes']
        # ids of classes in WildScenes
        selected_labels = [mi['cidx'][mi['classes'].index(cls)] for cls in classes]

        lidar_points = position(self.get_cloud(i, gravity_aligned=False))
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

            img_plane_points = ego_to_cam(lidar_points.T, E[:3, :3], E[:3, 3], K).T
            mask_img = get_only_in_img_mask(img_plane_points.T, seg_label_cam.shape[0], seg_label_cam.shape[1])
            img_plane_points = img_plane_points[mask_img]
            cam_points = lidar_points[mask_img].numpy()

            # colorize point cloud with values from segmentation image
            uv = img_plane_points[:, :2].numpy().astype(int)
            seg_label_cam = seg_label_cam[uv[:, 1], uv[:, 0]]

            points.append(cam_points)
            labels.append(seg_label_cam)

        points = np.concatenate(points)
        labels = np.concatenate(labels)
        colors = self.seg_label_to_color(labels)
        assert len(points) == len(colors)

        # mask out points with labels not in selected classes
        mask_labels = np.isin(labels, selected_labels)
        points = points[mask_labels]
        colors = colors[mask_labels]

        # gravity-aligned cloud
        pose_grav_aligned = self.get_initial_pose_on_heightmap(i)
        points = transform_cloud(points, pose_grav_aligned)

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
        :return: heightmap (2 x H x W), where 2 is the number of channels (z and mask)
        """
        if dir_name is None:
            dir_name = os.path.join(self.path, 'terrain', 'rigid')

        file_path = os.path.join(dir_name, f'{self.ids[i]}.npy')
        if cached and os.path.exists(file_path):
            hm_rigid = np.load(file_path)
        else:
            traj_points = self.get_footprint_traj_points(i, T_horizon=10.0)
            soft_classes = self.lss_cfg['soft_classes']
            rigid_classes = [c for c in WILDSCENES_METAINFO['classes'] if c not in soft_classes]
            seg_points, _ = self.get_semantic_cloud(i, classes=rigid_classes, vis=False)
            points = np.concatenate((seg_points, traj_points), axis=0)
            points = torch.as_tensor(points, dtype=torch.float32)
            hm_rigid = estimate_heightmap(points, d_max=self.dphys_cfg.d_max,
                                          grid_res=self.grid_res,
                                          h_max=self.dphys_cfg.h_max)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, hm_rigid.cpu().numpy())
        heightmap = torch.as_tensor(hm_rigid)
        return heightmap

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_images_data(i)
        control_ts, controls = self.get_controls(i)
        traj_ts, states = self.get_states_traj(i)
        Xs, Xds, Rs, Omegas = states
        hm_geom = self.get_geom_height_map(i)
        hm_terrain = self.get_terrain_height_map(i)
        pose0 = torch.as_tensor(self.get_initial_pose_on_heightmap(i), dtype=torch.float32)
        return (imgs, rots, trans, intrins, post_rots, post_trans,
                hm_geom, hm_terrain,
                control_ts, controls,
                pose0,
                traj_ts, Xs, Xds, Rs, Omegas)
