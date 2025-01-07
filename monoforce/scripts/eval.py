#!/usr/bin/env python

import sys
sys.path.append('../src/')
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
from monoforce.models.traj_predictor.traj_lstm import TrajLSTM
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.terrain_encoder.bevfusion import BEVFusion
from monoforce.models.terrain_encoder.voxelnet import VoxelNet
from monoforce.transformations import position
from monoforce.datasets.rough import ROUGH
from monoforce.models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, denormalize_img
from monoforce.utils import read_yaml, write_to_csv, append_to_csv, compile_data
from monoforce.losses import physics_loss, hm_loss
import matplotlib as mpl


np.random.seed(42)
torch.manual_seed(42)

def arg_parser():
    parser = argparse.ArgumentParser(description='Terrain encoder predictor input arguments')
    parser.add_argument('--robot', type=str, default='tradr', help='Robot name')
    parser.add_argument('--terrain_encoder', type=str, default='lss', help='Terrain encoder model')
    parser.add_argument('--terrain_encoder_path', type=str, default=None, help='Path to the LSS model')
    parser.add_argument('--traj_predictor', type=str, default='dphysics', help='Trajectory predictor model')
    parser.add_argument('--vis', action='store_true', help='Visualize the results')
    return parser.parse_args()


class Data(ROUGH):
    def __init__(self, path, lss_cfg=None, dphys_cfg=DPhysConfig(), is_train=True):
        super(Data, self).__init__(path, lss_cfg, dphys_cfg=dphys_cfg, is_train=is_train)

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_images_data(i)
        points = torch.as_tensor(position(self.get_cloud(i))).T
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
                traj_ts, Xs, Xds, Rs, Omegas,
                points)

class Eval:
    def __init__(self,
                 robot='marv',
                 terrain_encoder='lss',
                 terrain_encoder_path=None,
                 traj_predictor='dphysics'):
        self.device = 'cpu'  # for visualization purposes using CPU

        # load DPhys config
        self.dphys_cfg = DPhysConfig(robot=robot)
        self.traj_predictor = self.get_traj_pred(model=traj_predictor)

        # load LSS config
        self.lss_config = read_yaml(os.path.join('..', 'config/lss_cfg.yaml'))
        self.terrain_encoder = self.get_terrain_encoder(terrain_encoder_path, model=terrain_encoder)
        self.output_folder = f'./gen/eval/{robot}_{self.terrain_encoder.__class__.__name__}_{self.traj_predictor.__class__.__name__}'

        # load data
        self.loader = self.get_dataloader()

    def get_terrain_encoder(self, path, model='lss'):
        if model == 'lss':
            terrain_encoder = LiftSplatShoot(self.lss_config['grid_conf'],
                                             self.lss_config['data_aug_conf']).from_pretrained(path)
        elif model == 'bevfusion':
            terrain_encoder = BEVFusion(self.lss_config['grid_conf'],
                                        self.lss_config['data_aug_conf']).from_pretrained(path)
        elif model == 'voxelnet':
            terrain_encoder = VoxelNet(self.lss_config['grid_conf']).from_pretrained(path)
        else:
            raise ValueError(f'Invalid terrain encoder model: {model}. Supported: lss, bevfusion, voxelnet')
        terrain_encoder.to(self.device)
        terrain_encoder.eval()
        return terrain_encoder

    def predict_terrain(self, batch):
        model = self.terrain_encoder.__class__.__name__
        if model == 'LiftSplatShoot':
            imgs, rots, trans, intrins, post_rots, post_trans = batch[:6]
            img_inputs = (imgs, rots, trans, intrins, post_rots, post_trans)
            terrain = self.terrain_encoder(*img_inputs)
        elif model == 'BEVFusion':
            imgs, rots, trans, intrins, post_rots, post_trans = batch[:6]
            img_inputs = (imgs, rots, trans, intrins, post_rots, post_trans)
            points_inputs = batch[-1]
            terrain = self.terrain_encoder(img_inputs, points_inputs)
        elif model == 'VoxelNet':
            points_inputs = batch[-1]
            terrain = self.terrain_encoder(points_inputs)
        else:
            raise ValueError(f'Invalid terrain encoder model: {model}. Supported: LiftSplatShoot, BEVFusion, VoxelNet')
        return terrain

    def get_traj_pred(self, model='dphysics'):
        if model == 'dphysics':
            traj_predictor = DPhysics(self.dphys_cfg, device=self.device)
        elif model == 'traj_lstm':
            h = w = int(2 * self.dphys_cfg.d_max / self.dphys_cfg.grid_res)
            traj_predictor = TrajLSTM(state_features=6,
                                      control_features=2,
                                      heightmap_shape=(h, w)).from_pretrained('../config/weights/traj_lstm/lstm.pth')
        else:
            raise ValueError(f'Invalid trajectory predictor model: {model}. Supported: dphysics, traj_lstm')
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
        elif model == 'TrajLSTM':
            controls = batch[9]
            height = terrain['terrain']
            pose0 = batch[10]
            xyz0 = pose0[:, :3, 3]
            rpy0 = torch.as_tensor(Rotation.from_matrix(pose0[:, :3, :3]).as_euler('xyz'), dtype=xyz0.dtype).to(self.device)
            xyz_rpy0 = torch.cat([xyz0, rpy0], dim=-1)
            xyz_rpy_pred = self.traj_predictor(xyz_rpy0, controls, height)
            X_pred = xyz_rpy_pred[:, :, :3]
            states_pred = [X_pred, None, None, None]
        else:
            raise ValueError(f'Invalid model: {model}. Supported: DPhysics, TrajLSTM')
        return states_pred

    def get_dataloader(self):
        train_ds, val_ds = compile_data(lss_cfg=self.lss_config, dphys_cfg=self.dphys_cfg, Data=Data)
        loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        return loader

    def run(self, vis=False):
        # create output folder
        os.makedirs(self.output_folder, exist_ok=True)
        # write losses to output csv
        write_to_csv(f'{self.output_folder}/losses.csv', 'Image id,Terrain Loss,Physics Loss\n')

        with torch.no_grad():
            if vis:
                H, W = self.lss_config['data_aug_conf']['H'], self.lss_config['data_aug_conf']['W']
                cams = ['cam_front', 'cam_left', 'cam_rear', 'cam_right']

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
                 traj_ts, Xs, Xds, Rs, Omegas,
                 points) = batch
                states_gt = [Xs, Xds, Rs, Omegas]

                # terrain prediction
                terrain = self.predict_terrain(batch)
                height_pred, friction_pred = terrain['terrain'], terrain['friction']

                # evaluation losses
                loss_terrain = hm_loss(height_pred=height_pred[0, 0], height_gt=hm_terrain[0, 0], weights=hm_terrain[0, 1])
                states_pred = self.predict_states(terrain, batch)
                loss_physics = physics_loss(states_pred=states_pred, states_gt=states_gt, pred_ts=control_ts, gt_ts=traj_ts)

                # write losses to csv
                append_to_csv(f'{self.output_folder}/losses.csv',
                              f'{i:04d}.png, {loss_terrain.item():.4f},{loss_physics.item():.4f}\n')

                # visualizations
                if vis:
                    height_pred = height_pred[0, 0].cpu()
                    friction_pred = friction_pred[0, 0].cpu()
                    # get height map points
                    hm_points = torch.stack([x_grid, y_grid, height_pred], dim=-1)
                    hm_points = hm_points.view(-1, 3).T

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
                    plt.imshow(height_pred.T, origin='lower', cmap='jet', vmin=-1., vmax=1.)
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

                    plt.pause(0.01)
                    plt.draw()

                    plt.savefig(f'{self.output_folder}/{i:04d}.png')

            if vis:
                plt.close(fig)

def main():
    args = arg_parser()
    print(args)
    monoforce = Eval(robot=args.robot,
                     terrain_encoder=args.terrain_encoder,
                     terrain_encoder_path=args.terrain_encoder_path,
                     traj_predictor=args.traj_predictor)
    monoforce.run(vis=args.vis)


if __name__ == '__main__':
    main()
