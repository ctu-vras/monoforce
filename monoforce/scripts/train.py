#!/usr/bin/env python

import sys
sys.path.append('../src')
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from monoforce.models.terrain_encoder.utils import denormalize_img, ego_to_cam, get_only_in_img_mask
from monoforce.models.terrain_encoder.lss import load_model
from monoforce.models.dphysics import DPhysics
from monoforce.dphys_config import DPhysConfig
from monoforce.utils import read_yaml, write_to_yaml, str2bool, compile_data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import argparse


np.random.seed(42)
torch.manual_seed(42)


def arg_parser():
    parser = argparse.ArgumentParser(description='Train MonoForce model')
    parser.add_argument('--bsz', type=int, default=4, help='Batch size')
    parser.add_argument('--nepochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay')
    parser.add_argument('--robot', type=str, default='marv', help='Robot name')
    parser.add_argument('--lss_cfg_path', type=str, default='../config/lss_cfg.yaml', help='Path to LSS config')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--debug', type=str2bool, default=True, help='Debug mode: use small datasets')
    parser.add_argument('--vis', type=str2bool, default=False, help='Visualize training samples')
    parser.add_argument('--only_front_cam', type=str2bool, default=False, help='Use only front heightmap')
    parser.add_argument('--terrain_hm_weight', type=float, default=1.0, help='Weight for terrain heightmap loss')
    parser.add_argument('--phys_weight', type=float, default=0.1, help='Weight for physics loss')

    return parser.parse_args()


class Trainer:
    """
    Trainer for terrain encoder model

    Args:
    robot: str, robot name
    dphys_cfg: DPhysConfig, physical robot-terrain interaction configuration
    lss_cfg: dict, LSS model configuration
    bsz: int, batch size
    lr: float, learning rate
    weight_decay: float, weight decay
    nepochs: int, number of epochs
    pretrained_model_path: str, path to pretrained model
    log_dir: str, path to log directory
    debug: bool, debug mode: use small datasets
    vis: bool, visualize training samples
    terrain_hm_weight: float, weight for terrain heightmap loss
    only_front_cam: bool, use only front heightmap part for training
    """

    def __init__(self,
                 robot,
                 dphys_cfg,
                 lss_cfg,
                 bsz=1,
                 lr=1e-3,
                 weight_decay=1e-7,
                 nepochs=1000,
                 pretrained_model_path=None,
                 debug=False,
                 vis=False,
                 terrain_hm_weight=10.0,
                 phys_weight=1.0,
                 only_front_cam=False):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'rough'
        self.robot = robot
        assert robot in ['husky', 'tradr', 'tradr2', 'husky_oru', 'marv'], 'Unknown robot: %s' % robot
        self.dphys_cfg = dphys_cfg
        self.lss_cfg = lss_cfg

        self.nepochs = nepochs
        self.min_loss = np.inf
        self.min_train_loss = np.inf
        self.train_counter = 0
        self.val_counter = 0

        self.terrain_hm_weight = terrain_hm_weight
        self.phys_weight = phys_weight

        self.only_front_cam = only_front_cam

        self.train_loader, self.val_loader = self.create_dataloaders(bsz=bsz, debug=debug, vis=vis)
        self.terrain_encoder = load_model(modelf=pretrained_model_path, lss_cfg=self.lss_cfg, device=self.device)
        self.terrain_encoder.train()
        self.dphysics = DPhysics(dphys_cfg, device=self.device)

        self.optimizer = torch.optim.Adam(self.terrain_encoder.parameters(), lr=lr, weight_decay=weight_decay)

        self.log_dir = os.path.join('../config/tb_runs',
                                    f'{self.dataset}_{self.robot}/lss_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
        self.writer = SummaryWriter(log_dir=self.log_dir)
        # save configs to log dir
        write_to_yaml(dphys_cfg.__dict__, os.path.join(self.log_dir, 'dphys_cfg.yaml'))
        write_to_yaml(lss_cfg, os.path.join(self.log_dir, 'lss_cfg.yaml'))

    def create_dataloaders(self, bsz=1, debug=False, vis=False):
        # create dataset for LSS model training
        train_ds, val_ds = compile_data(dphys_cfg=self.dphys_cfg, lss_cfg=self.lss_cfg,
                                        small_data=debug, vis=vis,
                                        only_front_cam=self.only_front_cam)

        # create dataloaders
        train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bsz, shuffle=False)

        return train_loader, val_loader

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

    def physics_loss(self, heightmap, friction, control_ts, controls, traj_ts, states):
        # predict states
        states_pred, forces_pred = self.dphysics(z_grid=heightmap, controls=controls, friction=friction)

        # unpack states
        X, Xd, R, Omega = states
        X_pred, Xd_pred, R_pred, Omega_pred, _ = states_pred

        # find the closest timesteps in the trajectory to the ground truth timesteps
        ts_ids = torch.argmin(torch.abs(control_ts.unsqueeze(1) - traj_ts.unsqueeze(2)), dim=2)

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

    def epoch(self, train=True):
        loader = self.train_loader if train else self.val_loader
        counter = self.train_counter if train else self.val_counter

        if train:
            self.terrain_encoder.train()
        else:
            self.terrain_encoder.eval()

        max_grad_norm = 5.0
        epoch_losses = {'terrain': 0.0, 'phys': 0.0, 'total': 0.0}
        for batch in tqdm(loader, total=len(loader)):
            batch = [torch.as_tensor(b, dtype=torch.float32, device=self.device) for b in batch]
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_terrain,
             control_ts, controls,
             traj_ts, Xs, Xds, Rs, Omegas) = batch

            terrain, weights_terrain = hm_terrain[:, 0:1], hm_terrain[:, 1:2]

            if train:
                self.optimizer.zero_grad()

            # terrain encoder forward pass
            inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
            out = self.terrain_encoder(*inputs)
            terrain_pred, friction_pred = out['terrain'], out['friction']
            # rigid / terrain height map loss
            loss_terrain = self.terrain_hm_loss(terrain_pred, terrain, weights_terrain) if self.terrain_hm_weight > 0 else torch.tensor(0.0, device=self.device)

            # physics loss: difference between predicted and ground truth states
            loss_phys = self.physics_loss(heightmap=terrain_pred.squeeze(1), friction=friction_pred.squeeze(1),
                                          control_ts=control_ts, controls=controls,
                                          traj_ts=traj_ts, states=[Xs, Xds, Rs, Omegas]) if self.phys_weight > 0 else torch.tensor(0.0, device=self.device)
            # total loss
            loss = self.terrain_hm_weight * loss_terrain + self.phys_weight * loss_phys

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.terrain_encoder.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()

            epoch_losses['terrain'] += loss_terrain.item()
            epoch_losses['phys'] += loss_phys.item()
            epoch_losses['total'] += loss.item()

            counter += 1
            self.writer.add_scalar(f"{'train' if train else 'val'}/iter_loss_terrain", loss_terrain, counter)
            self.writer.add_scalar(f"{'train' if train else 'val'}/iter_loss_phys", loss_phys, counter)
            self.writer.add_scalar(f"{'train' if train else 'val'}/iter_loss", loss, counter)

        if len(loader) > 0:
            epoch_losses['terrain'] /= len(loader)
            epoch_losses['phys'] /= len(loader)
            epoch_losses['total'] /= len(loader)

        return epoch_losses, counter

    def train(self):
        for e in range(self.nepochs):
            # training epoch
            train_losses, self.train_counter = self.epoch(train=True)
            for k, v in train_losses.items():
                print('Epoch:', e, f'Train loss {k}:', v)
                self.writer.add_scalar(f'train/epoch_loss_{k}', v, e)

            if train_losses['total'] < self.min_train_loss:
                with torch.no_grad():
                    self.min_train_loss = train_losses['total']
                    print('Saving train model...')
                    self.terrain_encoder.eval()
                    torch.save(self.terrain_encoder.state_dict(), os.path.join(self.log_dir, 'train_lss.pt'))

                    # visualize training predictions
                    fig = self.vis_pred(self.train_loader)
                    self.writer.add_figure('train/prediction', fig, e)

            # validation epoch
            with torch.no_grad():
                val_losses, self.val_counter = self.epoch(train=False)
                for k, v in val_losses.items():
                    print('Epoch:', e, f'Val loss {k}:', v)
                    self.writer.add_scalar(f'val/epoch_loss_{k}', v, e)
                if val_losses['total'] < self.min_loss:
                    self.min_loss = val_losses['total']
                    print('Saving model...')
                    self.terrain_encoder.eval()
                    torch.save(self.terrain_encoder.state_dict(), os.path.join(self.log_dir, 'lss.pt'))

                    # visualize validation predictions
                    fig = self.vis_pred(self.val_loader)
                    self.writer.add_figure('val/prediction', fig, e)

    def vis_pred(self, loader):
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(341)
        ax2 = fig.add_subplot(342)
        ax3 = fig.add_subplot(343)
        ax4 = fig.add_subplot(344)
        ax5 = fig.add_subplot(345)
        ax6 = fig.add_subplot(346)
        ax7 = fig.add_subplot(347)
        ax8 = fig.add_subplot(348)

        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
        for ax in axes:
            ax.clear()

        # visualize training predictions
        with torch.no_grad():
            # unpack batch
            sample = loader.dataset[np.random.choice(len(loader.dataset))]
            batch = [torch.as_tensor(b[None], device=self.device) for b in sample]
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_terrain,
             ts_controls, controls,
             ts_traj, Xs, Xds, Rs, Omegas) = batch

            # predict height maps
            inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
            out = self.terrain_encoder(*inputs)
            terrain_pred, friction_pred = out['terrain'], out['friction']

            # predict states
            states_pred, _ = self.dphysics(z_grid=terrain_pred.squeeze(1), controls=controls, friction=friction_pred.squeeze(1))

            batch_i = 0
            terrain_pred = terrain_pred[batch_i, 0].cpu()
            terrain = hm_terrain[batch_i, 0].cpu()
            friction_pred = friction_pred[batch_i, 0].cpu()
            xyz_pred = states_pred[0][batch_i].cpu().numpy()
            xyz = Xs[batch_i].cpu().numpy()

            # get height map points
            z_grid = terrain_pred
            x_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
            y_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
            x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
            hm_points = torch.stack([x_grid, y_grid, z_grid], dim=-1)
            hm_points = hm_points.view(-1, 3).T

            # plot images with projected height map points
            inputs = [i.cpu() for i in inputs]
            imgs, rots, trans, intrins, post_rots, post_trans = inputs
            for imgi in range(imgs.shape[1])[:4]:
                ax = axes[imgi]
                img = imgs[batch_i, imgi]
                img = denormalize_img(img[:3])
                cam_pts = ego_to_cam(hm_points, rots[batch_i, imgi], trans[batch_i, imgi], intrins[batch_i, imgi])
                img_H, img_W = self.lss_cfg['data_aug_conf']['H'], self.lss_cfg['data_aug_conf']['W']
                mask_img = get_only_in_img_mask(cam_pts, img_H, img_W)
                plot_pts = post_rots[batch_i, imgi].matmul(cam_pts) + post_trans[batch_i, imgi].unsqueeze(1)
                ax.imshow(img)
                ax.scatter(plot_pts[0, mask_img], plot_pts[1, mask_img], s=1, c=hm_points[2, mask_img],
                           cmap='jet', vmin=-1.0, vmax=1.0)
                ax.axis('off')

            ax5.set_title('Prediction: Terrain')
            ax5.imshow(terrain_pred.T, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)

            ax6.set_title('Label: Terrain')
            ax6.imshow(terrain.T, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)

            ax7.set_title('Friction')
            ax7.imshow(friction_pred.T, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)

            ax8.set_title('Trajectories')
            ax8.plot(xyz[:, 0], xyz[:, 1], 'kx', label='GT')
            ax8.plot(xyz_pred[:, 0], xyz_pred[:, 1], 'r.', label='Pred')
            ax8.set_xlabel('X [m]')
            ax8.set_ylabel('Y [m]')
            ax8.grid()
            ax8.legend()
            # ax8.set_xlim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
            # ax8.set_ylim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
            ax8.axis('equal')

            return fig


def main():
    args = arg_parser()
    print(args)

    # load configs: DPhys
    dphys_cfg = DPhysConfig(robot=args.robot)
    # load configs: LSS
    lss_config_path = args.lss_cfg_path
    assert os.path.isfile(lss_config_path), 'LSS config file %s does not exist' % lss_config_path
    lss_cfg = read_yaml(lss_config_path)

    # create trainer
    trainer = Trainer(robot=args.robot,
                      dphys_cfg=dphys_cfg, lss_cfg=lss_cfg,
                      bsz=args.bsz, nepochs=args.nepochs,
                      lr=args.lr, weight_decay=args.weight_decay,
                      pretrained_model_path=args.pretrained_model_path,
                      terrain_hm_weight=args.terrain_hm_weight,
                      phys_weight=args.phys_weight,
                      debug=args.debug, vis=args.vis,
                      only_front_cam=args.only_front_cam)
    trainer.train()


if __name__ == '__main__':
    main()
