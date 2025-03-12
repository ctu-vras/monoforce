#!/usr/bin/env python

import sys
sys.path.append('../src/')
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation
import argparse
from monoforce.models.traj_predictor.dphys_config import DPhysConfig
from monoforce.models.traj_predictor.dphysics import DPhysics
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, denormalize_img
from monoforce.utils import read_yaml, write_to_csv, append_to_csv, compile_data, str2bool
from monoforce.losses import physics_loss, hm_loss
from monoforce.datasets import ROUGH, rough_seq_paths

np.random.seed(42)
torch.manual_seed(42)


def arg_parser():
    parser = argparse.ArgumentParser(description='Terrain encoder predictor input arguments')
    parser.add_argument('--seq', type=str, default='val', help='Data sequence')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--terrain_encoder', type=str, default='lss', help='Terrain encoder model')
    parser.add_argument('--terrain_encoder_path', type=str, default=None, help='Path to the LSS model')
    parser.add_argument('--traj_predictor', type=str, default='dphysics', help='Trajectory predictor model')
    parser.add_argument('--vis', type=str2bool, default=False, help='Visualize the results')
    return parser.parse_args()


class Eval:
    def __init__(self,
                 seq='val',
                 batch_size=1,
                 terrain_encoder='lss',
                 terrain_encoder_path=None,
                 traj_predictor='dphysics'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load DPhys config
        if seq in rough_seq_paths:
            robot = os.path.basename(seq).split('_')[0]
            robot = 'tradr' if robot == 'ugv' else 'marv'
        else:
            robot = 'marv'
        print(f'Robot: {robot}')
        self.dphys_cfg = DPhysConfig(robot=robot)
        self.traj_predictor = self.get_traj_pred(model=traj_predictor)

        # load LSS config
        self.lss_config = read_yaml(os.path.join('..', 'config/lss_cfg.yaml'))
        self.terrain_encoder = self.get_terrain_encoder(terrain_encoder_path, model=terrain_encoder)

        # load data
        self.loader = self.get_dataloader(batch_size=batch_size, seq=seq)

        # output folder to write evaluation results
        self.output_folder = (f'./gen/eval_{os.path.basename(seq)}/'
                              f'{robot}_{self.terrain_encoder.__class__.__name__}_'
                              f'{self.traj_predictor.__class__.__name__}')

    def get_terrain_encoder(self, path, model='lss'):
        if model == 'lss':
            terrain_encoder = LiftSplatShoot(self.lss_config['grid_conf'],
                                             self.lss_config['data_aug_conf']).from_pretrained(path)
        else:
            raise ValueError(f'Invalid terrain encoder model: {model}. Supported: lss')
        terrain_encoder.to(self.device)
        terrain_encoder.eval()
        return terrain_encoder

    def predict_terrain(self, batch):
        model = self.terrain_encoder.__class__.__name__
        if model == 'LiftSplatShoot':
            imgs, rots, trans, intrins, post_rots, post_trans = batch[:6]
            img_inputs = (imgs, rots, trans, intrins, post_rots, post_trans)
            terrain = self.terrain_encoder(*img_inputs)
        else:
            raise ValueError(f'Invalid terrain encoder model: {model}. Supported: LiftSplatShoot')
        return terrain

    def get_traj_pred(self, model='dphysics'):
        if model == 'dphysics':
            traj_predictor = DPhysics(self.dphys_cfg, device=self.device)
        else:
            raise ValueError(f'Invalid trajectory predictor model: {model}. Supported: dphysics')
        traj_predictor.to(self.device)
        traj_predictor.eval()
        return traj_predictor

    def predict_states(self, terrain, batch):
        model = self.traj_predictor.__class__.__name__
        if model == 'DPhysics':
            Xs, Xds, Rs, Omegas = batch[12:16]
            controls = batch[9]
            state0 = tuple([s[:, 0] for s in [Xs, Xds, Rs, Omegas]])
            height, friction = terrain['terrain'], terrain['friction']
            states_pred, _ = self.traj_predictor(z_grid=height.squeeze(1), state=state0,
                                                 controls=controls, friction=friction.squeeze(1))
        else:
            raise ValueError(f'Invalid model: {model}. Supported: DPhysics')
        return states_pred

    def get_dataloader(self, batch_size=1, seq='val'):
        if seq != 'val':
            print('Loading dataset from:', seq)
            val_ds = ROUGH(path=seq, lss_cfg=self.lss_config, dphys_cfg=self.dphys_cfg)
        else:
            _, val_ds = compile_data(lss_cfg=self.lss_config, dphys_cfg=self.dphys_cfg)
        loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return loader

    @torch.inference_mode()
    def run(self, vis=False):
        # create output folder
        os.makedirs(self.output_folder, exist_ok=True)
        # write losses to output csv
        write_to_csv(f'{self.output_folder}/losses.csv', 'Batch i,H_g loss,H_t loss,XYZ loss,Rot loss\n')

        H, W = self.lss_config['data_aug_conf']['H'], self.lss_config['data_aug_conf']['W']
        cams = ['cam_left', 'cam_front', 'cam_right', 'cam_rear']

        x_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
        y_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
        x_grid, y_grid = torch.meshgrid(x_grid, y_grid)

        fig, axes = plt.subplots(3, 4, figsize=(20, 16))
        for i, batch in enumerate(tqdm(self.loader)):
            batch = [t.to(self.device) for t in batch]
            # get a sample from the dataset
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_geom, hm_terrain,
             control_ts, controls,
             pose0,
             traj_ts, Xs, Xds, Rs, Omegas) = batch
            states_gt = [Xs, Xds, Rs, Omegas]

            # terrain prediction
            terrain = self.predict_terrain(batch)
            H_t_pred, H_g_pred, H_diff_pred, Friction_pred = terrain['terrain'], terrain['geom'], terrain['diff'], \
            terrain['friction']

            # terrain and geom heightmap losses
            loss_geom = hm_loss(height_pred=H_g_pred[:, 0], height_gt=hm_geom[:, 0], weights=hm_geom[:, 1])
            loss_terrain = hm_loss(height_pred=H_t_pred[:, 0], height_gt=hm_terrain[:, 0], weights=hm_terrain[:, 1])

            # trajectory prediction loss: xyz and rotation
            states_pred = self.predict_states(terrain, batch)
            loss_xyz, loss_rot = physics_loss(states_pred=states_pred, states_gt=states_gt,
                                              pred_ts=control_ts, gt_ts=traj_ts,
                                              gamma=1.0, rotation_loss=True)

            # write losses to csv
            append_to_csv(f'{self.output_folder}/losses.csv',
                          f'{i:04d}, {loss_geom.item()},{loss_terrain.item()},{loss_xyz.item()},{loss_rot.item()}\n')

            # visualizations
            H_g_pred = H_g_pred[0, 0].cpu()
            H_diff_pred = H_diff_pred[0, 0].cpu()
            H_t_pred = H_t_pred[0, 0].cpu()
            Friction_pred = Friction_pred[0, 0].cpu()
            # get height map points
            hm_points = torch.stack([x_grid, y_grid, H_t_pred], dim=-1)
            hm_points = hm_points.view(-1, 3).T

            batch = [t.to('cpu') for t in batch]
            # get a sample from the dataset
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_geom, hm_terrain,
             control_ts, controls,
             pose0,
             traj_ts, Xs, Xds, Rs, Omegas) = batch
            states_gt = [Xs, Xds, Rs, Omegas]

            # clear axis
            for ax in axes.flatten():
                ax.clear()
            plt.suptitle(f'Terrain Loss: {loss_terrain.item():.3f}, Traj Loss: {loss_xyz.item() + loss_rot.item():.3f}')
            for imgi, img in enumerate(imgs[0]):
                ax = axes[0, imgi]

                cam_pts = ego_to_cam(hm_points, rots[0, imgi], trans[0, imgi], intrins[0, imgi])
                mask = get_only_in_img_mask(cam_pts, H, W)
                plot_pts = post_rots[0, imgi].matmul(cam_pts) + post_trans[0, imgi].unsqueeze(1)
                showimg = denormalize_img(img)

                ax.imshow(showimg)
                ax.scatter(plot_pts[0, mask], plot_pts[1, mask],
                           # c=Friction_pred.view(-1)[terrain_mask][mask],
                           c=hm_points[2, mask],
                           s=2, alpha=0.8, cmap='jet', vmin=-1, vmax=1.)
                ax.axis('off')
                # camera name as text on image
                ax.text(0.5, 0.9, cams[imgi].replace('_', ' '),
                        horizontalalignment='center', verticalalignment='top',
                        transform=ax.transAxes, fontsize=10)

            # plot geom heightmap
            axes[1, 0].set_title('Geom Height')
            axes[1, 0].imshow(H_g_pred.T, origin='lower', cmap='jet', vmin=-1., vmax=1.)
            axes[1, 0].axis('off')

            # plot height diff heightmap
            axes[1, 1].set_title('Height Difference')
            axes[1, 1].imshow(H_diff_pred.T, origin='lower', cmap='jet', vmin=-1., vmax=1.)
            axes[1, 1].axis('off')

            # plot terrain heightmap
            axes[1, 2].set_title('Terrain Height')
            axes[1, 2].imshow(H_t_pred.T, origin='lower', cmap='jet', vmin=-1., vmax=1.)
            axes[1, 2].axis('off')

            # plot friction map
            axes[1, 3].set_title('Friction')
            axes[1, 3].imshow(Friction_pred.T, origin='lower', cmap='jet', vmin=0., vmax=1.)
            axes[1, 3].axis('off')

            # plot control inputs
            axes[2, 0].plot(control_ts[0], controls[0, :, 0], c='g', label='v(t)')
            axes[2, 0].plot(control_ts[0], controls[0, :, 1], c='b', label='w(t)')
            axes[2, 0].grid()
            axes[2, 0].set_xlabel('Time [s]')
            axes[2, 0].set_ylabel('Control [m/s]')
            axes[2, 0].legend()

            # plot trajectories: Roll, Pitch, Yaw
            rpy = Rotation.from_matrix(states_pred[2][0].cpu()).as_euler('xyz')
            rpy_gt = Rotation.from_matrix(states_gt[2][0].cpu()).as_euler('xyz')
            axes[2, 1].plot(control_ts[0], rpy[:, 0], 'r', label='Pred Roll')
            axes[2, 1].plot(control_ts[0], rpy[:, 1], 'g', label='Pred Pitch')
            axes[2, 1].plot(control_ts[0], rpy[:, 2], 'b', label='Pred Yaw')
            axes[2, 1].plot(traj_ts[0], rpy_gt[:, 0], 'r--', label='Roll')
            axes[2, 1].plot(traj_ts[0], rpy_gt[:, 1], 'g--', label='Pitch')
            axes[2, 1].plot(traj_ts[0], rpy_gt[:, 2], 'b--', label='Yaw')
            axes[2, 1].grid()
            axes[2, 1].set_xlabel('Time [s]')
            axes[2, 1].set_ylabel('Angle [rad]')
            axes[2, 1].set_ylim(-np.pi / 2., np.pi / 2.)
            # axes[2, 1].legend()

            # plot trajectories: XY
            axes[2, 2].plot(states_pred[0][0, :, 0].cpu(), states_pred[0][0, :, 1].cpu(), 'r', label='Pred Traj')
            axes[2, 2].plot(states_gt[0][0, :, 0], states_gt[0][0, :, 1], 'k', label='GT Traj')
            axes[2, 2].set_xlim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
            axes[2, 2].set_ylim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
            axes[2, 2].grid()
            axes[2, 2].set_xlabel('x [m]')
            axes[2, 2].set_ylabel('y [m]')
            axes[2, 2].legend()

            # plot trajectories: Z
            axes[2, 3].plot(control_ts[0], states_pred[0][0, :, 2].cpu(), 'r', label='Pred Traj')
            axes[2, 3].plot(traj_ts[0], states_gt[0][0, :, 2], 'k', label='GT Traj')
            axes[2, 3].grid()
            axes[2, 3].set_xlabel('Time [s]')
            axes[2, 3].set_ylabel('z [m]')
            axes[2, 3].set_ylim(-self.dphys_cfg.h_max, self.dphys_cfg.h_max)
            axes[2, 3].legend()

            if vis:
                plt.pause(0.01)
                plt.draw()
            plt.savefig(f'{self.output_folder}/{i:04d}.png')
        plt.close(fig)


def main():
    args = arg_parser()
    print(args)
    monoforce = Eval(seq=args.seq,
                     batch_size=args.batch_size,
                     terrain_encoder=args.terrain_encoder,
                     terrain_encoder_path=args.terrain_encoder_path,
                     traj_predictor=args.traj_predictor)
    monoforce.run(vis=args.vis)


if __name__ == '__main__':
    main()
