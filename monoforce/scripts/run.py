#!/usr/bin/env python

import sys
sys.path.append('../src')
import os
import numpy as np
import torch
from PIL import Image
from mayavi import mlab
import argparse
from monoforce.models.traj_predictor.dphys_config import DPhysConfig
from monoforce.models.traj_predictor.dphysics import DPhysics, generate_control_inputs
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.terrain_encoder.utils import denormalize_img, normalize_img, img_transform, sample_augmentation
from monoforce.utils import read_yaml, load_calib
from monoforce.vis import visualize_imgs


def arg_parser():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='Run MonoForce')
    parser.add_argument('--lss_cfg_path', type=str,
                        default=os.path.join(base_path, 'config/lss_cfg.yaml'), help='Path to the LSS config file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the LSS model')
    parser.add_argument('--img-paths', type=str, required=True, nargs='+', help='Paths to the input RGB images')
    parser.add_argument('--calibration-path', type=str, required=True, help='Path to the calibration files')
    parser.add_argument('--cameras', type=str, nargs='+', default=None, help='Camera names')

    return parser.parse_args()


class MonoForce:
    def __init__(self, imgs_path, calib_path,
                 cameras=None,
                 lss_cfg_path=os.path.join('..', 'config/lss_cfg.yaml'),
                 model_path=os.path.join('..', 'config/weights/lss/lss.pt')):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.imgs_path = imgs_path
        self.calib_path = calib_path

        # load DPhys config
        self.dphys_cfg = DPhysConfig(robot='tradr')
        self.dphys_cfg.traj_sim_time = 8.0

        # load LSS config
        self.lss_config_path = lss_cfg_path
        assert os.path.isfile(self.lss_config_path), 'LSS config file %s does not exist' % self.lss_config_path
        self.lss_config = read_yaml(self.lss_config_path)
        self.dphysics = DPhysics(self.dphys_cfg, device=self.device)

        self.model_path = model_path
        self.terrain_encoder = LiftSplatShoot(self.lss_config['grid_conf'],
                                              self.lss_config['data_aug_conf']).from_pretrained(self.model_path)
        self.terrain_encoder.to(self.device)

        # load calibration
        self.calib = load_calib(calib_path=self.calib_path)
        self.cameras = self.get_cameras() if cameras is None else cameras
        assert len(self.cameras) > 0, 'No camera calibration found in path %s' % self.calib_path

    def poses_from_states(self, states):
        xyz = states[0].squeeze().cpu().numpy()
        Rs = states[2].squeeze().cpu().numpy()
        poses = np.stack([np.eye(4) for _ in range(len(xyz))])
        poses[:, :3, :3] = Rs
        poses[:, :3, 3] = xyz
        return poses

    def predict_states(self, z_grid):
        T, dt = self.dphys_cfg.traj_sim_time, self.dphys_cfg.dt
        controls, _ = generate_control_inputs(n_trajs=z_grid.shape[0],
                                              v_range=(self.dphys_cfg.vel_max / 2., self.dphys_cfg.vel_max),
                                              w_range=(-self.dphys_cfg.omega_max, self.dphys_cfg.omega_max),
                                              time_horizon=T, dt=dt)
        controls = controls.to(self.device)
        states, forces = self.dphysics(z_grid, controls=controls)
        return states, forces

    def get_cameras(self):
        cams_yaml = os.listdir(os.path.join(self.calib_path, 'cameras'))
        cams = [cam.replace('.yaml', '') for cam in cams_yaml]
        if 'camera_up' in cams:
            cams.remove('camera_up')
        return sorted(cams)

    def get_preprocessed_data(self):
        imgs = []
        rots = []
        trans = []
        post_rots = []
        post_trans = []
        intrins = []
        for cam, img_path in zip(self.cameras, self.imgs_path):
            img = Image.open(img_path)
            K = self.calib[cam]['camera_matrix']['data']
            K = np.asarray(K, dtype=np.float32).reshape((3, 3))
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = sample_augmentation(self.lss_config)
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
        inputs = [torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                  torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans)]
        inputs = [torch.as_tensor(i, dtype=torch.float32, device=self.device) for i in inputs]

        return inputs

    def run(self):
        # load data
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_preprocessed_data()

        # draw input images
        imgs_vis = [np.asarray(denormalize_img(img)) for img in imgs]
        visualize_imgs(imgs_vis, names=self.cameras)

        # get heightmap prediction
        with torch.no_grad():
            inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
            inputs = [torch.as_tensor(i[None]) for i in inputs]
            out = self.terrain_encoder(*inputs)
            z_grid = out['terrain'].squeeze(1)

            # predict trajectory and interaction forces
            states, forces = self.predict_states(z_grid)
            poses = self.poses_from_states(states)
            print('Predicted poses shape: %s' % str(poses.shape))
            F_springs = forces[0].squeeze().cpu().numpy()
            print('Predicted forces shape: %s' % str(F_springs.shape))
            robot_points0 = np.asarray(self.dphys_cfg.robot_points)
            print('Robot contact points shape: %s' % str(robot_points0.shape))

        # visualize: - heightmap, - robot poses, - robot contact points, - interaction forces
        mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
        z_grid = z_grid.squeeze().cpu().numpy()
        h, w = z_grid.shape
        x_grid, y_grid = np.mgrid[-h//2:h//2, -w//2:w//2] * self.dphys_cfg.grid_res
        mlab.surf(x_grid, y_grid, z_grid, colormap='terrain', opacity=0.8)
        mlab.plot3d(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], color=(0, 0, 0), line_width=2.0)
        visu_robot = mlab.points3d(robot_points0[:, 0], robot_points0[:, 1], robot_points0[:, 2],
                                   color=(0, 0, 1), scale_factor=0.1)
        visu_forces = mlab.quiver3d(robot_points0[:, 0], robot_points0[:, 1], robot_points0[:, 2],
                                    F_springs[0, :, 0], F_springs[0, :, 1], F_springs[0, :, 2],
                                    line_width=1.0, scale_factor=0.001)
        for i in range(len(poses)):
            # robot-terrain contact points
            robot_points = robot_points0 @ poses[i, :3, :3].T + poses[i, :3, 3:4].T
            robot_points[..., 2] += 0.132
            visu_robot.mlab_source.set(x=robot_points[:, 0], y=robot_points[:, 1], z=robot_points[:, 2])
            visu_forces.mlab_source.set(x=robot_points[:, 0], y=robot_points[:, 1], z=robot_points[:, 2],
                                        u=F_springs[i, :, 0], v=F_springs[i, :, 1], w=F_springs[i, :, 2])
            mlab.view(azimuth=i/10, elevation=60, distance=20)
            # mlab pause
            if i % 10 == 0:
                os.makedirs('./gen', exist_ok=True)
                mlab.savefig(f'./gen/{i//10:04d}.png')
        mlab.show()


def main():
    args = arg_parser()
    print('Loading MonoForce with the following arguments:')
    print(args)
    monoforce = MonoForce(imgs_path=args.img_paths,
                          calib_path=args.calibration_path,
                          cameras=args.cameras,
                          lss_cfg_path=args.lss_cfg_path,
                          model_path=args.model_path)
    monoforce.run()


if __name__ == '__main__':
    main()
