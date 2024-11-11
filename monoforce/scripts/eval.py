#!/usr/bin/env python

import sys
sys.path.append('../src/')
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import argparse
from datetime import datetime
from monoforce.dphys_config import DPhysConfig
from monoforce.models.dphysics import DPhysics
from monoforce.models.terrain_encoder.lss import load_model
from monoforce.datasets.rough import ROUGH, rough_seq_paths
from monoforce.models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, denormalize_img
from monoforce.utils import read_yaml, write_to_csv, append_to_csv
import matplotlib as mpl


def arg_parser():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='Terrain encoder predictor input arguments')
    parser.add_argument('--robot', type=str, default='tradr2', help='Robot name')
    parser.add_argument('--lss_cfg_path', type=str,
                        default=os.path.join(base_path, 'config/lss_cfg.yaml'), help='Path to the LSS config file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the LSS model')
    parser.add_argument('--seq_i', type=int, default=0, help='Data sequence index')
    return parser.parse_args()


class Evaluation:
    def __init__(self,
                 robot='marv',
                 lss_cfg_path=os.path.join('..', 'config/lss_cfg.yaml'),
                 model_path=os.path.join('..', 'config/weights/lss/lss.pt'),
                 seq_i=0):
        self.device = 'cpu'  # for visualization purposes using CPU

        # load DPhys config
        self.dphys_cfg = DPhysConfig(robot=robot)
        self.dphysics = DPhysics(self.dphys_cfg, device=self.device)

        # load LSS config
        self.lss_config_path = lss_cfg_path
        assert os.path.isfile(self.lss_config_path), 'LSS config file %s does not exist' % self.lss_config_path
        self.lss_config = read_yaml(self.lss_config_path)
        self.model_path = model_path
        self.terrain_encoder = load_model(self.model_path, self.lss_config, device=self.device)

        # load dataset
        self.path = rough_seq_paths[seq_i]
        self.ds = ROUGH(path=self.path, lss_cfg=self.lss_config, dphys_cfg=self.dphys_cfg)
        self.loader = torch.utils.data.DataLoader(self.ds, batch_size=1, shuffle=False)

        # create output folder
        self.output_folder = f'./gen_{os.path.basename(self.path)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(self.output_folder, exist_ok=True)
        # write losses to output csv
        write_to_csv(f'{self.output_folder}/losses.csv', 'Image id,Terrain Loss,Physics Loss\n')

    def terrain_hm_loss(self, height_pred, height_gt, weights=None):
        assert height_pred.shape == height_gt.shape, 'Height prediction and ground truth must have the same shape'
        if weights is None:
            weights = torch.ones_like(height_gt)
        assert weights.shape == height_gt.shape, 'Weights and height ground truth must have the same shape'

        # remove nan values
        mask_valid = ~torch.isnan(height_gt)
        height_gt = height_gt[mask_valid]
        height_pred = height_pred[mask_valid]
        weights = weights[mask_valid]

        # compute weighted loss
        loss = torch.nn.functional.mse_loss(height_pred * weights, height_gt * weights, reduction='mean')
        assert not torch.isnan(loss), 'Terrain Loss is nan'

        return loss

    def physics_loss(self, states_pred, states_gt, pred_ts, gt_ts):
        # unpack the states
        X, Xd, R, Omega = states_gt[:4]
        X_pred, Xd_pred, R_pred, Omega_pred = states_pred[:4]

        # find the closest timesteps in the trajectory to the ground truth timesteps
        ts_ids = torch.argmin(torch.abs(pred_ts.unsqueeze(1) - gt_ts.unsqueeze(2)), dim=2)

        # get the predicted states at the closest timesteps to the ground truth timesteps
        batch_size = X.shape[0]
        X_pred_gt_ts = X_pred[torch.arange(batch_size).unsqueeze(1), ts_ids]

        # remove nan values
        mask_valid = ~torch.isnan(X_pred_gt_ts)
        X_pred_gt_ts = X_pred_gt_ts[mask_valid]
        X = X[mask_valid]
        loss = torch.nn.functional.mse_loss(X_pred_gt_ts, X)
        assert not torch.isnan(loss), 'Physics Loss is nan'

        return loss

    def run(self):
        with torch.no_grad():
            H, W = self.lss_config['data_aug_conf']['H'], self.lss_config['data_aug_conf']['W']
            cams = self.ds.camera_names

            n_rows, n_cols = 2, int(np.ceil(len(cams) / 2) + 3)
            img_h, img_w = self.lss_config['data_aug_conf']['final_dim']
            ratio = img_h / img_w
            fig = plt.figure(figsize=(n_cols * 5, n_rows * 4 * ratio))
            gs = mpl.gridspec.GridSpec(n_rows, n_cols)
            gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

            x_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
            y_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
            x_grid, y_grid = torch.meshgrid(x_grid, y_grid)

            for i, batch in enumerate(tqdm(self.loader)):
                batch = [t.to(self.device) for t in batch]
                # get a sample from the dataset
                (imgs, rots, trans, intrins, post_rots, post_trans,
                 hm_terrain,
                 control_ts, controls,
                 traj_ts, Xs, Xds, Rs, Omegas) = batch

                # terrain prediction
                inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
                out = self.terrain_encoder(*inputs)
                terrain_pred, friction_pred = out['terrain'], out['friction']

                # evaluation losses
                terrain_loss = self.terrain_hm_loss(height_pred=terrain_pred[0, 0], height_gt=hm_terrain[0, 0])
                states_gt = [Xs, Xds, Rs, Omegas]
                states_pred, _ = self.dphysics(z_grid=terrain_pred.squeeze(1), controls=controls, friction=friction_pred.squeeze(1))
                physics_loss = self.physics_loss(states_pred, states_gt, pred_ts=control_ts, gt_ts=traj_ts)

                # visualizations
                terrain_pred = terrain_pred[0, 0].cpu()
                friction_pred = friction_pred[0, 0].cpu()

                # get height map points
                z_grid = terrain_pred
                hm_points = torch.stack([x_grid, y_grid, z_grid], dim=-1)
                hm_points = hm_points.view(-1, 3).T

                plt.clf()
                plt.suptitle(f'Terrain Loss: {terrain_loss.item():.4f}, Physics Loss: {physics_loss.item():.4f}')
                for imgi, img in enumerate(imgs[0]):
                    cam_pts = ego_to_cam(hm_points, rots[0, imgi], trans[0, imgi], intrins[0, imgi])
                    mask = get_only_in_img_mask(cam_pts, H, W)
                    plot_pts = post_rots[0, imgi].matmul(cam_pts) + post_trans[0, imgi].unsqueeze(1)

                    ax = plt.subplot(gs[imgi // int(np.ceil(len(cams) / 2)), imgi % int(np.ceil(len(cams) / 2))])
                    showimg = denormalize_img(img)

                    plt.imshow(showimg)
                    plt.scatter(plot_pts[0, mask], plot_pts[1, mask], c=friction_pred.view(-1)[mask],
                                s=2, alpha=0.8, cmap='jet', vmin=0., vmax=1.)
                    plt.axis('off')
                    # camera name as text on image
                    plt.text(0.5, 0.9, cams[imgi].replace('_', ' '),
                             horizontalalignment='center', verticalalignment='top',
                             transform=ax.transAxes, fontsize=10)

                # plot terrain heightmap
                plt.subplot(gs[:, -3:-2])
                plt.title('Terrain Height')
                plt.imshow(terrain_pred.T, origin='lower', cmap='jet', vmin=-1., vmax=1.)
                plt.axis('off')
                plt.colorbar()

                # plot friction map
                plt.subplot(gs[:, -2:-1])
                plt.title('Friction')
                plt.imshow(friction_pred.T, origin='lower', cmap='jet', vmin=0., vmax=1.)
                plt.axis('off')
                plt.colorbar()

                # plot trajectories
                plt.subplot(gs[:, -1:])
                plt.plot(states_pred[0].squeeze()[:, 0], states_pred[0].squeeze()[:, 1], 'r.', label='Pred Traj')
                plt.plot(states_gt[0].squeeze()[:, 0], states_gt[0].squeeze()[:, 1], 'kx', label='GT Traj')
                plt.xlim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
                plt.ylim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
                plt.grid()
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')
                plt.legend()

                plt.pause(0.01)
                plt.draw()

                plt.savefig(f'{self.output_folder}/{i:04d}.png')
                append_to_csv(f'{self.output_folder}/losses.csv',
                              f'{i:04d}.png, {terrain_loss.item():.4f},{physics_loss.item():.4f}\n')

            plt.close(fig)


def main():
    args = arg_parser()
    print(args)
    monoforce = Evaluation(robot=args.robot,
                           lss_cfg_path=args.lss_cfg_path,
                           model_path=args.model_path,
                           seq_i=args.seq_i)
    monoforce.run()


if __name__ == '__main__':
    main()
