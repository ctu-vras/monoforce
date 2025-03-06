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
import argparse
from monoforce.models.traj_predictor.dphys_config import DPhysConfig
from monoforce.models.traj_predictor.dphysics import DPhysics
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, denormalize_img
from monoforce.utils import read_yaml, write_to_csv, append_to_csv, compile_data
from monoforce.losses import physics_loss, hm_loss
from monoforce.datasets import ROUGH, rough_seq_paths


np.random.seed(42)
torch.manual_seed(42)

def arg_parser():
    parser = argparse.ArgumentParser(description='Terrain encoder predictor input arguments')
    parser.add_argument('--robot', type=str, default='tradr', help='Robot name')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--terrain_encoder', type=str, default='lss', help='Terrain encoder model')
    parser.add_argument('--terrain_encoder_path', type=str, default=None, help='Path to the LSS model')
    parser.add_argument('--traj_predictor', type=str, default='dphysics', help='Trajectory predictor model')
    parser.add_argument('--vis', action='store_true', help='Visualize the results')
    return parser.parse_args()


class Eval:
    def __init__(self,
                 robot='marv',
                 batch_size=1,
                 terrain_encoder='lss',
                 terrain_encoder_path=None,
                 traj_predictor='dphysics'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load DPhys config
        self.dphys_cfg = DPhysConfig(robot=robot)
        self.traj_predictor = self.get_traj_pred(model=traj_predictor)

        # load LSS config
        self.lss_config = read_yaml(os.path.join('..', 'config/lss_cfg.yaml'))
        self.terrain_encoder = self.get_terrain_encoder(terrain_encoder_path, model=terrain_encoder)
        self.output_folder = f'./gen/eval_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}/{robot}_{self.terrain_encoder.__class__.__name__}_{self.traj_predictor.__class__.__name__}'

        # load data
        self.loader = self.get_dataloader(batch_size=batch_size)

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

    def get_dataloader(self, batch_size=1):
        # val_ds = ROUGH(path=rough_seq_paths[1], lss_cfg=self.lss_config, dphys_cfg=self.dphys_cfg)
        _, val_ds = compile_data(lss_cfg=self.lss_config, dphys_cfg=self.dphys_cfg)
        loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return loader

    def run(self, vis=False):
        # create output folder
        os.makedirs(self.output_folder, exist_ok=True)
        # write losses to output csv
        write_to_csv(f'{self.output_folder}/losses.csv', 'Batch i,H_g loss,H_t loss,XYZ loss,Rot loss\n')

        with torch.no_grad():
            if vis:
                fig = plt.figure(figsize=(20, 8))
                # remove whitespace around the figure
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                H, W = self.lss_config['data_aug_conf']['H'], self.lss_config['data_aug_conf']['W']
                cams = ['cam_left', 'cam_front', 'cam_right', 'cam_rear']

                x_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
                y_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
                x_grid, y_grid = torch.meshgrid(x_grid, y_grid)

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
                H_t_pred, H_g_pred, Friction_pred = terrain['terrain'], terrain['geom'], terrain['friction']

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
                if vis:
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

                    plt.clf()
                    plt.suptitle(f'Terrain Loss: {loss_terrain.item():.3f}, Traj Loss: {loss_xyz.item() + loss_rot.item():.3f}')
                    for imgi, img in enumerate(imgs[0]):
                        cam_pts = ego_to_cam(hm_points, rots[0, imgi], trans[0, imgi], intrins[0, imgi])
                        mask = get_only_in_img_mask(cam_pts, H, W)
                        plot_pts = post_rots[0, imgi].matmul(cam_pts) + post_trans[0, imgi].unsqueeze(1)

                        ax = plt.subplot(2, 4, imgi + 1)
                        showimg = denormalize_img(img)

                        plt.imshow(showimg)
                        plt.scatter(plot_pts[0, mask], plot_pts[1, mask],
                                    # c=Friction_pred.view(-1)[terrain_mask][mask],
                                    c=hm_points[2, mask],
                                    s=2, alpha=0.8, cmap='jet', vmin=-1, vmax=1.)
                        plt.axis('off')
                        # camera name as text on image
                        plt.text(0.5, 0.9, cams[imgi].replace('_', ' '),
                                 horizontalalignment='center', verticalalignment='top',
                                 transform=ax.transAxes, fontsize=10)

                    # plot terrain heightmap
                    plt.subplot(2, 4, 5)
                    plt.title('Terrain Height')
                    plt.imshow(H_t_pred.T, origin='lower', cmap='jet', vmin=-1., vmax=1.)
                    plt.axis('off')
                    plt.colorbar()

                    # plot friction map
                    plt.subplot(2, 4, 6)
                    plt.title('Friction')
                    plt.imshow(Friction_pred.T, origin='lower', cmap='jet', vmin=0., vmax=1.)
                    plt.axis('off')
                    plt.colorbar()

                    # plot trajectories: XY
                    plt.subplot(2, 4, 7)
                    plt.plot(states_pred[0][0, :, 0].cpu(), states_pred[0][0, :, 1].cpu(), 'r.', label='Pred Traj')
                    plt.plot(states_gt[0][0, :, 0], states_gt[0][0, :, 1], 'kx', label='GT Traj')
                    plt.xlim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
                    plt.ylim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
                    plt.grid()
                    plt.xlabel('x [m]')
                    plt.ylabel('y [m]')
                    plt.xlim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
                    plt.ylim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
                    plt.legend()

                    # plot trajectories: Z
                    plt.subplot(2, 4, 8)
                    plt.plot(control_ts[0], states_pred[0][0, :, 2].cpu(), 'r.', label='Pred Traj')
                    plt.plot(traj_ts[0], states_gt[0][0, :, 2], 'kx', label='GT Traj')
                    plt.grid()
                    plt.xlabel('Time [s]')
                    plt.ylabel('z [m]')
                    plt.ylim(-self.dphys_cfg.h_max, self.dphys_cfg.h_max)
                    plt.legend()

                    plt.pause(0.01)
                    plt.draw()
                    plt.savefig(f'{self.output_folder}/{i:04d}.png')

            if vis:
                plt.close(fig)

def main():
    args = arg_parser()
    print(args)
    monoforce = Eval(robot=args.robot,
                     batch_size=args.batch_size,
                     terrain_encoder=args.terrain_encoder,
                     terrain_encoder_path=args.terrain_encoder_path,
                     traj_predictor=args.traj_predictor)
    monoforce.run(vis=args.vis)


if __name__ == '__main__':
    main()
