#!/usr/bin/env python

import sys
sys.path.append('../src')
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from monoforce.models.terrain_encoder.utils import denormalize_img, ego_to_cam, get_only_in_img_mask
from eval import Evaluator
from monoforce.utils import read_yaml, write_to_yaml, str2bool, compile_data
from monoforce.losses import hm_loss, physics_loss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import argparse


torch.autograd.set_detect_anomaly(True)


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
    parser.add_argument('--traj_sim_time', type=float, default=5.0, help='Trajectory simulation time')
    parser.add_argument('--dphys_grid_res', type=float, default=0.4, help='DPhys grid resolution')

    return parser.parse_args()


class Trainer(Evaluator):
    def __init__(self,
                 batch_size=1,
                 n_epochs=1000,
                 lr=1e-3,
                 pretrained_terrain_encoder_path=None,
                 geom_weight=1.0,
                 terrain_weight=1.0,
                 phys_weight=1.0):
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
        self.optimizer = torch.optim.Adam(self.terrain_encoder.parameters(),
                                          lr=lr, betas=(0.8, 0.999), weight_decay=1e-7)
        # load datasets
        self.train_loader, self.val_loader = self.create_dataloaders(debug=False, vis=False)

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
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def compute_losses(self, batch):
        # unpack batch
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain,
         control_ts, controls,
         pose0,
         traj_ts, Xs, Xds, Rs, Omegas) = batch

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
            states_gt = [Xs, Xds, Rs, Omegas]
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
                    # fig = self.vis_pred(self.train_loader)
                    # self.writer.add_figure('train/prediction', fig, e)

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
                        # fig = self.vis_pred(self.val_loader)
                        # self.writer.add_figure('val/prediction', fig, e)


def main():
    args = arg_parser()
    print(args)

    trainer = Trainer(batch_size=args.batch_size,
                      lr=args.lr,
                      n_epochs=args.n_epochs,
                      pretrained_terrain_encoder_path=args.pretrained_terrain_encoder_path,
                      geom_weight=args.geom_weight,
                      terrain_weight=args.terrain_weight,
                      phys_weight=args.phys_weight
                      )
    trainer.train()


if __name__ == '__main__':
    main()