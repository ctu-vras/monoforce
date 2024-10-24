#!/usr/bin/env python

import sys
sys.path.append('../src')
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import torch
import argparse
from datetime import datetime
from monoforce.dphys_config import DPhysConfig
from monoforce.models.dphysics import DPhysics
from monoforce.models.terrain_encoder.lss import load_model
from monoforce.datasets.robingas import RobinGas, robingas_seq_paths
from monoforce.models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, denormalize_img
from monoforce.utils import read_yaml


def arg_parser():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('monoforce_demos', 'monoforce')

    parser = argparse.ArgumentParser(description='Terrain encoder predictor input arguments')
    parser.add_argument('--robot', type=str, default='tradr2', help='Robot name')
    parser.add_argument('--lss_cfg_path', type=str,
                        default=os.path.join(base_path, 'config/lss_cfg.yaml'), help='Path to the LSS config file')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(base_path, 'config/weights/lss/lss.pt'), help='Path to the LSS model')
    parser.add_argument('--seq_i', type=int, default=0, help='Data sequence index')
    return parser.parse_args()


class Predictor:
    def __init__(self,
                 robot='marv',
                 lss_cfg_path=os.path.join('..', 'config/lss_cfg.yaml'),
                 model_path=os.path.join('..', 'config/weights/lss/lss.pt'),
                 seq_i=0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.path = robingas_seq_paths[robot][seq_i]
        self.ds = RobinGas(path=self.path, lss_cfg=self.lss_config, dphys_cfg=self.dphys_cfg)

    def poses_from_states(self, states):
        xyz = states[0].squeeze().cpu().numpy()
        Rs = states[2].squeeze().cpu().numpy()
        poses = np.stack([np.eye(4) for _ in range(len(xyz))])
        poses[:, :3, :3] = Rs
        poses[:, :3, 3] = xyz
        return poses

    def run(self):
        with torch.no_grad():
            H, W = self.lss_config['data_aug_conf']['H'], self.lss_config['data_aug_conf']['W']
            cams = self.ds.camera_names

            n_rows, n_cols = 2, int(np.ceil(len(cams) / 2) + 2)
            img_h, img_w = self.lss_config['data_aug_conf']['final_dim']
            ratio = img_h / img_w
            fig = plt.figure(figsize=(n_cols * 4, n_rows * 4 * ratio))
            gs = mpl.gridspec.GridSpec(n_rows, n_cols)
            gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

            x_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
            y_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
            x_grid, y_grid = torch.meshgrid(x_grid, y_grid)

            output_folder = f'./gen_{os.path.basename(self.path)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            for i in tqdm(range(len(self.ds))):
                # get a sample from the dataset
                sample = self.ds[i]
                (imgs, rots, trans, intrins, post_rots, post_trans,
                 hm_geom, hm_terrain,
                 control_ts, controls,
                 traj_ts, Xs, Xds, Rs, Omegas) = sample

                # terrain prediction
                inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
                inputs = [torch.as_tensor(i[None], device=self.device) for i in inputs]
                out = self.terrain_encoder(*inputs)
                (height_pred_geom, height_pred_terrain,
                 height_pred_diff, friction_pred) = (out['geom'], out['terrain'],
                                                     out['diff'], out['friction'])
                # print(height_pred_terrain.shape, friction_pred.shape)

                # # dynamics prediction: robot's trajectory and terrain interaction forces
                # controls = torch.as_tensor(controls[None], device=self.device)
                # states, forces = self.dphysics(height_pred_terrain.squeeze(1), controls=controls)
                # print(states[0].shape, forces[0].shape)
                # poses = self.poses_from_states(states)

                # visualizations
                batch_i = 0
                height_pred_geom = height_pred_geom[batch_i, 0].cpu()
                height_pred_terrain = height_pred_terrain[batch_i, 0].cpu()
                friction_pred = friction_pred[batch_i, 0].cpu()

                # get height map points
                z_grid = height_pred_terrain
                hm_points = torch.stack([x_grid, y_grid, z_grid], dim=-1)
                hm_points = hm_points.view(-1, 3).T

                plt.clf()
                for imgi, img in enumerate(imgs):
                    cam_pts = ego_to_cam(hm_points, rots[imgi], trans[imgi], intrins[imgi])
                    mask = get_only_in_img_mask(cam_pts, H, W)
                    plot_pts = post_rots[imgi].matmul(cam_pts) + post_trans[imgi].unsqueeze(1)

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

                # plot height maps
                plt.subplot(gs[:, -2:-1])
                plt.title('Terrain Height')
                plt.imshow(height_pred_geom.T, origin='lower', cmap='jet', vmin=-1., vmax=1.)
                plt.axis('off')
                plt.colorbar()

                plt.subplot(gs[:, -1:])
                plt.title('Friction')
                plt.imshow(friction_pred.T, origin='lower', cmap='jet', vmin=0., vmax=1.)
                plt.axis('off')
                plt.colorbar()

                plt.pause(0.01)
                plt.draw()

                os.makedirs(output_folder, exist_ok=True)
                plt.savefig(f'{output_folder}/{i:04d}.png')

            plt.close(fig)

            # create a video from the generated images using ffmpeg
            os.system(f'ffmpeg -y -r 10 -i {output_folder}/%04d.png -vcodec libx264 -pix_fmt yuv420p -crf 25 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {output_folder}/{os.path.basename(self.path)}.mp4')


def main():
    args = arg_parser()
    print(args)
    monoforce = Predictor(robot=args.robot,
                          lss_cfg_path=args.lss_cfg_path,
                          model_path=args.model_path,
                          seq_i=args.seq_i)
    monoforce.run()


if __name__ == '__main__':
    main()
