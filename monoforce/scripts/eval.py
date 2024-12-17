#!/usr/bin/env python

import sys
sys.path.append('../src/')
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import argparse
from monoforce.dphys_config import DPhysConfig
from monoforce.models.dphysics import DPhysics
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.terrain_encoder.bevfusion import BEVFusion
from monoforce.transformations import transform_cloud, position
from monoforce.datasets.rough import ROUGH, rough_seq_paths
from monoforce.models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, denormalize_img
from monoforce.utils import read_yaml, write_to_csv, append_to_csv
from monoforce.losses import physics_loss, hm_loss
import matplotlib as mpl


def arg_parser():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description='Terrain encoder predictor input arguments')
    parser.add_argument('--robot', type=str, default='tradr', help='Robot name')
    parser.add_argument('--lss_cfg_path', type=str,
                        default=os.path.join(base_path, 'config/lss_cfg.yaml'), help='Path to the LSS config file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the LSS model')
    parser.add_argument('--seq_i', type=int, default=0, help='Data sequence index')
    parser.add_argument('--vis', action='store_true', help='Visualize the results')
    parser.add_argument('--save', action='store_true', help='Save the results')
    return parser.parse_args()


class Fusion(ROUGH):
    def __init__(self, path, lss_cfg=None, dphys_cfg=DPhysConfig(), is_train=True):
        super(Fusion, self).__init__(path, lss_cfg, dphys_cfg=dphys_cfg, is_train=is_train)

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_images_data(i)
        points = torch.as_tensor(position(self.get_cloud(i))).T
        control_ts, controls = self.get_controls(i)
        traj_ts, states = self.get_states_traj(i)
        Xs, Xds, Rs, Omegas = states
        hm_geom = self.get_terrain_height_map(i)
        hm_terrain = self.get_geom_height_map(i)

        return (imgs, rots, trans, intrins, post_rots, post_trans,
                hm_geom, hm_terrain,
                control_ts, controls,
                traj_ts, Xs, Xds, Rs, Omegas,
                points)

class Evaluation:
    def __init__(self,
                 robot='marv',
                 lss_cfg_path=os.path.join('..', 'config/lss_cfg.yaml'),
                 model_path=None,
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
        self.terrain_encoder = LiftSplatShoot(self.lss_config['grid_conf'],
                                              self.lss_config['data_aug_conf']).from_pretrained(self.model_path)
        self.terrain_encoder.to(self.device)
        # self.terrain_encoder = BEVFusion(self.lss_config['grid_conf'],
        #                                  self.lss_config['data_aug_conf']).from_pretrained(self.model_path)
        # load dataset
        self.path = rough_seq_paths[seq_i]
        self.ds = ROUGH(path=self.path, lss_cfg=self.lss_config, dphys_cfg=self.dphys_cfg, is_train=False)
        # self.ds = Fusion(path=self.path, lss_cfg=self.lss_config, dphys_cfg=self.dphys_cfg, is_train=False)
        self.loader = torch.utils.data.DataLoader(self.ds, batch_size=1, shuffle=False)

    def run(self, vis=False, save=False):
        if save:
            # create output folder
            self.output_folder = f'./gen_{os.path.basename(self.path)}'
            os.makedirs(self.output_folder, exist_ok=True)
            # write losses to output csv
            write_to_csv(f'{self.output_folder}/losses.csv', 'Image id,Terrain Loss,Physics Loss\n')

        with torch.no_grad():
            H, W = self.lss_config['data_aug_conf']['H'], self.lss_config['data_aug_conf']['W']
            cams = self.ds.camera_names

            n_rows, n_cols = 2, int(np.ceil(len(cams) / 2)) + 4
            img_h, img_w = self.lss_config['data_aug_conf']['final_dim']
            ratio = img_h / img_w
            fig = plt.figure(figsize=(n_cols * 5, n_rows * ratio * 4))
            gs = mpl.gridspec.GridSpec(n_rows, n_cols)
            gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

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
                # (imgs, rots, trans, intrins, post_rots, post_trans,
                #  hm_geom, hm_terrain,
                #  control_ts, controls,
                #  pose0,
                #  traj_ts, Xs, Xds, Rs, Omegas,
                #  points) = batch

                # terrain prediction
                img_inputs = (imgs, rots, trans, intrins, post_rots, post_trans)
                out = self.terrain_encoder(*img_inputs)
                # out = self.terrain_encoder(img_inputs, points)
                terrain_pred, friction_pred = out['terrain'], out['friction']

                # # grount-truth terrain
                # terrain_pred = hm_terrain[:, 0:1]
                # friction_pred = torch.ones_like(terrain_pred)

                # evaluation losses
                loss_terrain = hm_loss(height_pred=terrain_pred[0, 0], height_gt=hm_terrain[0, 0], weights=hm_terrain[0, 1])
                states_gt = [Xs, Xds, Rs, Omegas]
                state0 = tuple([s[:, 0] for s in states_gt])
                states_pred, _ = self.dphysics(z_grid=terrain_pred.squeeze(1), state=state0,
                                               controls=controls, friction=friction_pred.squeeze(1))
                loss_physics = physics_loss(states_pred=states_pred, states_gt=states_gt, pred_ts=control_ts, gt_ts=traj_ts)

                # visualizations
                terrain_pred = terrain_pred[0, 0].cpu()
                friction_pred = friction_pred[0, 0].cpu()

                # get height map points
                hm_points = torch.stack([x_grid, y_grid, terrain_pred], dim=-1)
                hm_points = hm_points.view(-1, 3).T

                # terrain_mask = hm_terrain[0, 1].cpu().bool().flatten()
                # hm_points = hm_points[:, terrain_mask]

                plt.clf()
                plt.suptitle(f'Terrain Loss: {loss_terrain.item():.4f}, Physics Loss: {loss_physics.item():.4f}')
                for imgi, img in enumerate(imgs[0]):
                    cam_pts = ego_to_cam(hm_points, rots[0, imgi], trans[0, imgi], intrins[0, imgi])
                    mask = get_only_in_img_mask(cam_pts, H, W)
                    plot_pts = post_rots[0, imgi].matmul(cam_pts) + post_trans[0, imgi].unsqueeze(1)

                    ax = plt.subplot(gs[imgi // int(np.ceil(len(cams) / 2)), imgi % int(np.ceil(len(cams) / 2))])
                    showimg = denormalize_img(img)

                    plt.imshow(showimg)
                    plt.scatter(plot_pts[0, mask], plot_pts[1, mask],
                                # c=friction_pred.view(-1)[terrain_mask][mask],
                                c=hm_points[2, mask],
                                s=2, alpha=0.8, cmap='jet', vmin=-1, vmax=1.)
                    plt.axis('off')
                    # camera name as text on image
                    plt.text(0.5, 0.9, cams[imgi].replace('_', ' '),
                             horizontalalignment='center', verticalalignment='top',
                             transform=ax.transAxes, fontsize=10)

                # plot terrain heightmap
                plt.subplot(gs[:, 2])
                plt.title('Terrain Height')
                plt.imshow(terrain_pred.T, origin='lower', cmap='jet', vmin=-1., vmax=1.)
                plt.axis('off')
                plt.colorbar()

                # plot friction map
                plt.subplot(gs[:, 3])
                plt.title('Friction')
                plt.imshow(friction_pred.T, origin='lower', cmap='jet', vmin=0., vmax=1.)
                plt.axis('off')
                plt.colorbar()

                # plot trajectories: XY
                plt.subplot(gs[:, 4])
                plt.plot(states_pred[0].squeeze()[:, 0], states_pred[0].squeeze()[:, 1], 'r.', label='Pred Traj')
                plt.plot(states_gt[0].squeeze()[:, 0], states_gt[0].squeeze()[:, 1], 'kx', label='GT Traj')
                plt.xlim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
                plt.ylim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
                plt.grid()
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')
                plt.xlim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
                plt.ylim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
                plt.legend()

                # plot trajectories: Z
                plt.subplot(gs[:, 5])
                plt.plot(control_ts.squeeze(), states_pred[0].squeeze()[:, 2], 'r.', label='Pred Traj')
                plt.plot(traj_ts.squeeze(), states_gt[0].squeeze()[:, 2], 'kx', label='GT Traj')
                plt.grid()
                plt.xlabel('Time [s]')
                plt.ylabel('z [m]')
                plt.ylim(-self.dphys_cfg.h_max, self.dphys_cfg.h_max)
                plt.legend()

                if vis:
                    plt.pause(0.01)
                    plt.draw()

                if save:
                    plt.savefig(f'{self.output_folder}/{i:04d}.png')
                    append_to_csv(f'{self.output_folder}/losses.csv',
                                  f'{i:04d}.png, {loss_terrain.item():.4f},{loss_physics.item():.4f}\n')

            plt.close(fig)


def main():
    args = arg_parser()
    print(args)
    monoforce = Evaluation(robot=args.robot,
                           lss_cfg_path=args.lss_cfg_path,
                           model_path=args.model_path,
                           seq_i=args.seq_i)
    monoforce.run(vis=args.vis, save=args.save)


if __name__ == '__main__':
    main()
