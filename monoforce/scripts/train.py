#!/usr/bin/env python

import sys
sys.path.append('../src')
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from eval import Evaluator
from monoforce.models.terrain_encoder.utils import denormalize_img, ego_to_cam, get_only_in_img_mask
from monoforce.models.physics_engine.utils.environment import make_x_y_grids
from monoforce.utils import str2bool, compile_data
from monoforce.losses import hm_loss, physics_loss


def arg_parser():
    parser = argparse.ArgumentParser(description='Train MonoForce model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--pretrained_terrain_encoder_path', type=str, default=None,
                        help='Path to pretrained terrain encoder')
    parser.add_argument('--debug', type=str2bool, default=True, help='Debug mode: use small datasets')
    parser.add_argument('--vis', type=str2bool, default=False, help='Visualize training samples')
    parser.add_argument('--geom_weight', type=float, default=1.0, help='Weight for geometry loss')
    parser.add_argument('--terrain_weight', type=float, default=1.0, help='Weight for terrain heightmap loss')
    parser.add_argument('--phys_weight', type=float, default=1.0, help='Weight for physics loss')
    parser.add_argument('--dphys_grid_res', type=float, default=0.4, help='DPhys grid resolution')

    return parser.parse_args()


class Trainer(Evaluator):
    def __init__(self,
                 batch_size=1,
                 n_epochs=1000,
                 lr=1e-3,
                 weight_decay=1e-7,
                 pretrained_terrain_encoder_path=None,
                 geom_weight=1.0,
                 terrain_weight=1.0,
                 phys_weight=1.0,
                 debug=False,
                 vis=False):
        super(Trainer, self).__init__(batch_size=batch_size,
                                      pretrained_terrain_encoder_path=pretrained_terrain_encoder_path)
        self.n_epochs = n_epochs
        self.min_val_loss = np.inf
        self.min_train_loss = np.inf
        self.train_counter = 0
        self.val_counter = 0

        # weights for losses
        self.geom_weight = geom_weight
        self.terrain_weight = terrain_weight
        self.phys_weight = phys_weight

        # define optimizer
        self.optimizer = torch.optim.Adam(self.terrain_encoder.parameters(), lr=lr, weight_decay=weight_decay)

        # load datasets
        self.train_loader, self.val_loader = self.create_dataloaders(debug=debug, vis=vis)

        # tensorboard logging
        self.dataset = 'rough'
        self.terrain_encoder_model = 'lss'
        self.log_dir = os.path.join('../config/tb_runs/',
                                    f'{self.dataset}/'
                                    f'{self.terrain_encoder_model}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def create_dataloaders(self, debug=False, vis=False):
        # create dataset for LSS model training
        train_ds, val_ds = compile_data(small_data=debug, vis=vis)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, drop_last=True)
        return train_loader, val_loader

    def compute_losses(self, batch):
        # unpack batch
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain,
         control_ts, controls,
         traj_ts, xs, xds, qs, omegas, thetas) = batch

        # terrain encoder forward pass
        terrain = self.predict_terrain(batch)

        # geometry loss: difference between predicted and ground truth height maps
        if self.geom_weight > 0:
            loss_geom = hm_loss(terrain['geom'], hm_geom[:, 0:1], hm_geom[:, 1:2])
        else:
            loss_geom = torch.tensor(0.0, device=self.device)

        # rigid / terrain height map loss
        if self.terrain_weight > 0:
            loss_terrain = hm_loss(terrain['terrain'], hm_terrain[:, 0:1], hm_terrain[:, 1:2])
        else:
            loss_terrain = torch.tensor(0.0, device=self.device)

        # physics loss: difference between predicted and ground truth states
        if self.phys_weight > 0:
            # predict trajectory
            states_gt = [xs, xds, qs, omegas, thetas]
            states_pred = self.predict_states(terrain, batch)
            # compute physics loss
            loss_phys = physics_loss(states_pred=[states_pred.x.permute(1, 0, 2)], states_gt=states_gt,
                                     pred_ts=control_ts, gt_ts=traj_ts)
        else:
            loss_phys = torch.tensor(0.0, device=self.device)

        return loss_geom, loss_terrain, loss_phys

    def epoch(self, train=True):
        loader = self.train_loader if train else self.val_loader
        counter = self.train_counter if train else self.val_counter

        if train:
            self.terrain_encoder.train()
        else:
            self.terrain_encoder.eval()

        max_grad_norm = 1.0
        epoch_losses = {'geom': 0.0, 'terrain': 0.0, 'phys': 0.0, 'total': 0.0}
        for batch in tqdm(loader, total=len(loader)):
            if train:
                self.optimizer.zero_grad()

            batch = [torch.as_tensor(b, dtype=torch.float32, device=self.device) for b in batch]
            loss_geom, loss_terrain, loss_phys = self.compute_losses(batch)
            loss = self.geom_weight * loss_geom + self.terrain_weight * loss_terrain + self.phys_weight * loss_phys

            if torch.isnan(loss).item():
                torch.save(self.terrain_encoder.state_dict(), os.path.join(self.log_dir, 'train.pth'))
                raise ValueError('Loss is NaN')

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.terrain_encoder.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()

            epoch_losses['geom'] += loss_geom.item()
            epoch_losses['terrain'] += loss_terrain.item()
            epoch_losses['phys'] += loss_phys.item()
            epoch_losses['total'] += (loss_geom + loss_terrain + loss_phys).item()

            counter += 1
            self.writer.add_scalar(f"{'train' if train else 'val'}/iter_loss_geom", loss_geom.item(), counter)
            self.writer.add_scalar(f"{'train' if train else 'val'}/iter_loss_terrain", loss_terrain.item(), counter)
            self.writer.add_scalar(f"{'train' if train else 'val'}/iter_loss_phys", loss_phys.item(), counter)
            self.writer.add_scalar(f"{'train' if train else 'val'}/iter_loss_total", loss.item(), counter)

        if len(loader) > 0:
            for k, v in epoch_losses.items():
                epoch_losses[k] /= len(loader)

        return epoch_losses, counter

    def train(self):
        for e in range(self.n_epochs):
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
                    torch.save(self.terrain_encoder.state_dict(), os.path.join(self.log_dir, 'train.pth'))

                    # visualize training predictions
                    fig = self.vis_pred(self.train_loader)
                    self.writer.add_figure('train/prediction', fig, e)

            # validation epoch
            with torch.inference_mode():
                with torch.no_grad():
                    val_losses, self.val_counter = self.epoch(train=False)
                    for k, v in val_losses.items():
                        print('Epoch:', e, f'Val loss {k}:', v)
                        self.writer.add_scalar(f'val/epoch_loss_{k}', v, e)

                    if val_losses['total'] < self.min_val_loss:
                        self.min_val_loss = val_losses['total']
                        print('Saving model...')
                        self.terrain_encoder.eval()
                        torch.save(self.terrain_encoder.state_dict(), os.path.join(self.log_dir, 'val.pth'))

                        # visualize validation predictions
                        fig = self.vis_pred(self.val_loader)
                        self.writer.add_figure('val/prediction', fig, e)
    
    @torch.no_grad()
    def vis_pred(self, loader):
        # get a batch from the loader
        batch = next(iter(loader))
        batch = [torch.as_tensor(b, dtype=torch.float32, device=self.device) for b in batch]

        # predict height maps and states
        terrain = self.predict_terrain(batch)
        states_pred = self.predict_states(terrain, batch)

        # unpack batch
        sample = [b[0].cpu() for b in batch]
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain,
         control_ts, controls,
         traj_ts, xs, xds, qs, omegas, thetas) = sample

        geom_pred = terrain['geom'][0, 0].cpu()
        diff_pred = terrain['diff'][0, 0].cpu()
        terrain_pred = terrain['terrain'][0, 0].cpu()
        friction_pred = terrain['friction'][0, 0].cpu()
        Xs_pred = states_pred.x[:, 0].cpu()
        Xs_pred_grid = (Xs_pred[:, :2] + self.world_config.max_coord) / self.world_config.grid_res
        Xs_grid = (xs[:, :2] + self.world_config.max_coord) / self.world_config.grid_res

        # get height map points
        z_grid = terrain_pred
        x_grid, y_grid = make_x_y_grids(max_coord=self.world_config.max_coord,
                                        grid_res=self.world_config.grid_res,
                                        num_robots=1)
        hm_points = torch.stack([x_grid.squeeze(0), y_grid.squeeze(0), z_grid], dim=-1)
        hm_points = hm_points.view(-1, 3).T

        # plot images with projected height map points
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        img_H, img_W = self.lss_config['data_aug_conf']['H'], self.lss_config['data_aug_conf']['W']
        for imgi in range(len(imgs))[:4]:
            ax = axes[0, imgi]
            img = imgs[imgi]
            img = denormalize_img(img[:3])

            cam_pts = ego_to_cam(hm_points, rots[imgi], trans[imgi], intrins[imgi])
            mask_img = get_only_in_img_mask(cam_pts, img_H, img_W)
            plot_pts = post_rots[imgi].matmul(cam_pts) + post_trans[imgi].unsqueeze(1)

            cam_pts_Xs = ego_to_cam(xs[:, :3].T, rots[imgi], trans[imgi], intrins[imgi])
            mask_img_Xs = get_only_in_img_mask(cam_pts_Xs, img_H, img_W)
            plot_pts_Xs = post_rots[imgi].matmul(cam_pts_Xs) + post_trans[imgi].unsqueeze(1)

            cam_pts_Xs_pred = ego_to_cam(Xs_pred[:, :3].T, rots[imgi], trans[imgi], intrins[imgi])
            mask_img_Xs_pred = get_only_in_img_mask(cam_pts_Xs_pred, img_H, img_W)
            plot_pts_Xs_pred = post_rots[imgi].matmul(cam_pts_Xs_pred) + post_trans[imgi].unsqueeze(1)

            ax.imshow(img)
            ax.scatter(plot_pts[0, mask_img], plot_pts[1, mask_img], s=1, c=hm_points[2, mask_img],
                       cmap='jet', vmin=-1.0, vmax=1.0)
            ax.scatter(plot_pts_Xs[0, mask_img_Xs], plot_pts_Xs[1, mask_img_Xs], c='k', s=1)
            ax.scatter(plot_pts_Xs_pred[0, mask_img_Xs_pred], plot_pts_Xs_pred[1, mask_img_Xs_pred], c='r', s=1)
            ax.axis('off')

        axes[1, 0].set_title('Prediction: Terrain')
        axes[1, 0].imshow(terrain_pred, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)
        axes[1, 0].scatter(Xs_pred_grid[:, 0], Xs_pred_grid[:, 1], c='r', s=1)
        axes[1, 0].scatter(Xs_grid[:, 0], Xs_grid[:, 1], c='k', s=1)

        axes[1, 1].set_title('Label: Terrain')
        axes[1, 1].imshow(hm_terrain[0], origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)
        axes[1, 1].scatter(Xs_pred_grid[:, 0], Xs_pred_grid[:, 1], c='r', s=1)
        axes[1, 1].scatter(Xs_grid[:, 0], Xs_grid[:, 1], c='k', s=1)

        axes[1, 2].set_title('Friction')
        axes[1, 2].imshow(friction_pred, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)
        axes[1, 2].scatter(Xs_pred_grid[:, 0], Xs_pred_grid[:, 1], c='r', s=1)
        axes[1, 2].scatter(Xs_grid[:, 0], Xs_grid[:, 1], c='k', s=1)

        axes[1, 3].set_title('Trajectories XY')
        axes[1, 3].plot(xs[:, 0], xs[:, 1], c='k', label='GT')
        axes[1, 3].plot(Xs_pred[:, 0], Xs_pred[:, 1], c='r', label='Pred')
        axes[1, 3].set_xlabel('X [m]')
        axes[1, 3].set_ylabel('Y [m]')
        axes[1, 3].set_xlim(-self.world_config.max_coord, self.world_config.max_coord)
        axes[1, 3].set_ylim(-self.world_config.max_coord, self.world_config.max_coord)
        axes[1, 3].grid()
        axes[1, 3].legend()

        axes[2, 0].set_title('Prediction: Geom')
        axes[2, 0].imshow(geom_pred, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)
        axes[2, 0].scatter(Xs_pred_grid[:, 0], Xs_pred_grid[:, 1], c='r', s=5)
        axes[2, 0].scatter(Xs_grid[:, 0], Xs_grid[:, 1], c='k', s=1)

        axes[2, 1].set_title('Label: Geom')
        axes[2, 1].imshow(hm_geom[0], origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)
        axes[2, 1].scatter(Xs_pred_grid[:, 0], Xs_pred_grid[:, 1], c='r', s=5)
        axes[2, 1].scatter(Xs_grid[:, 0], Xs_grid[:, 1], c='k', s=1)

        axes[2, 2].set_title('Height diff')
        axes[2, 2].imshow(diff_pred, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)
        axes[2, 2].scatter(Xs_pred_grid[:, 0], Xs_pred_grid[:, 1], c='r', s=5)
        axes[2, 2].scatter(Xs_grid[:, 0], Xs_grid[:, 1], c='k', s=1)

        axes[2, 3].set_title('Trajectories Z')
        axes[2, 3].plot(traj_ts, xs[:, 2], 'k', label='GT')
        axes[2, 3].plot(control_ts, Xs_pred[:, 2], c='r', label='Pred')
        axes[2, 3].set_xlabel('Time [s]')
        axes[2, 3].set_ylabel('Z [m]')
        axes[2, 3].set_ylim(-1.0, 1.0)
        axes[2, 3].grid()
        axes[2, 3].legend()

        return fig


def main():
    args = arg_parser()
    print(args)

    trainer = Trainer(batch_size=args.batch_size,
                      lr=args.lr,
                      n_epochs=args.n_epochs,
                      pretrained_terrain_encoder_path=args.pretrained_terrain_encoder_path,
                      geom_weight=args.geom_weight,
                      terrain_weight=args.terrain_weight,
                      phys_weight=args.phys_weight,
                      debug=args.debug,
                      vis=args.vis)
    trainer.train()


if __name__ == '__main__':
    main()