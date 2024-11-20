#!/usr/bin/env python

import sys
sys.path.append('../src')
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from monoforce.models.terrain_encoder.utils import denormalize_img, ego_to_cam, get_only_in_img_mask
from monoforce.models.terrain_encoder.lss import load_lss_model
from monoforce.models.terrain_encoder.bevfusion import BEVFusion, LidarBEV
from monoforce.models.dphysics import DPhysics
from monoforce.dphys_config import DPhysConfig
from monoforce.datasets.rough import ROUGH
from monoforce.utils import read_yaml, write_to_yaml, str2bool, compile_data, position
from monoforce.transformations import transform_cloud
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import argparse


np.random.seed(42)
torch.manual_seed(42)


def arg_parser():
    parser = argparse.ArgumentParser(description='Train MonoForce model')
    parser.add_argument('--model', type=str, default='lss', help='Model to train: lss, bevfusion, lidarbev')
    parser.add_argument('--bsz', type=int, default=4, help='Batch size')
    parser.add_argument('--nepochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay')
    parser.add_argument('--robot', type=str, default='marv', help='Robot name')
    parser.add_argument('--lss_cfg_path', type=str, default='../config/lss_cfg.yaml', help='Path to LSS config')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--debug', type=str2bool, default=True, help='Debug mode: use small datasets')
    parser.add_argument('--vis', type=str2bool, default=True, help='Visualize training samples')
    parser.add_argument('--terrain_weight', type=float, default=1.0, help='Weight for terrain heightmap loss')
    parser.add_argument('--phys_weight', type=float, default=0.1, help='Weight for physics loss')

    return parser.parse_args()


class TrainerCore:
    """
    Trainer for terrain encoder model

    Args:
    dphys_cfg: DPhysConfig, physical robot-terrain interaction configuration
    lss_cfg: dict, LSS model configuration
    model: str, model to train: lss, bevfusion, lidarbev
    bsz: int, batch size
    lr: float, learning rate
    weight_decay: float, weight decay
    nepochs: int, number of epochs
    pretrained_model_path: str, path to pretrained model
    debug: bool, debug mode: use small datasets
    vis: bool, visualize training samples
    terrain_weight: float, weight for terrain heightmap loss
    phys_weight: float, weight for physics loss
    """

    def __init__(self,
                 dphys_cfg,
                 lss_cfg,
                 model,
                 bsz=1,
                 lr=1e-3,
                 weight_decay=1e-7,
                 nepochs=1000,
                 pretrained_model_path=None,
                 debug=False,
                 vis=False,
                 terrain_weight=1.0,
                 phys_weight=0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'rough'
        self.model = model
        self.dphys_cfg = dphys_cfg
        self.lss_cfg = lss_cfg

        self.nepochs = nepochs
        self.min_loss = np.inf
        self.min_train_loss = np.inf
        self.train_counter = 0
        self.val_counter = 0

        self.terrain_weight = terrain_weight
        self.phys_weight = phys_weight

        # models and optimizer
        self.terrain_encoder = None
        self.dphysics = DPhysics(dphys_cfg, device=self.device)
        
        # optimizer
        self.optimizer = None
        
        # dataloaders
        self.train_loader = None
        self.val_loader = None

        self.log_dir = os.path.join('../config/tb_runs/',
                                    f'{self.dataset}/{self.model}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def create_dataloaders(self, bsz=1, debug=False, vis=False, Data=ROUGH):
        # create dataset for LSS model training
        train_ds, val_ds = compile_data(dphys_cfg=self.dphys_cfg, lss_cfg=self.lss_cfg,
                                        small_data=debug, vis=vis, Data=Data)
        # create dataloaders: making sure all elemts in a batch are tensors
        def collate_fn(batch):
            def to_tensor(item):
                if isinstance(item, np.ndarray):
                    return torch.tensor(item)
                elif isinstance(item, (list, tuple)):
                    return [to_tensor(i) for i in item]
                elif isinstance(item, dict):
                    return {k: to_tensor(v) for k, v in item.items()}
                return item  # Return as is if it's already a tensor or unsupported type

            batch = [to_tensor(item) for item in batch]
            return torch.utils.data.default_collate(batch)

        train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=bsz, shuffle=False, collate_fn=collate_fn)

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

    def physics_loss(self, states_pred, states_gt, pred_ts, gt_ts):
        # unpack states
        X, Xd, R, Omega = states_gt
        X_pred, Xd_pred, R_pred, Omega_pred, _ = states_pred

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

    def compute_losses(self, batch):
        loss_terrain = torch.tensor(0.0, device=self.device)
        loss_phys = torch.tensor(0.0, device=self.device)
        return loss_terrain, loss_phys

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
            if train:
                self.optimizer.zero_grad()

            batch = [torch.as_tensor(b, dtype=torch.float32, device=self.device) for b in batch]
            loss_terrain, loss_phys = self.compute_losses(batch)
            loss = self.terrain_weight * loss_terrain + self.phys_weight * loss_phys

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
        # save configs to log dir
        write_to_yaml(self.dphys_cfg.__dict__, os.path.join(self.log_dir, 'dphys_cfg.yaml'))
        write_to_yaml(self.lss_cfg, os.path.join(self.log_dir, 'lss_cfg.yaml'))

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
                    if val_losses['total'] < self.min_loss:
                        self.min_loss = val_losses['total']
                        print('Saving model...')
                        self.terrain_encoder.eval()
                        torch.save(self.terrain_encoder.state_dict(), os.path.join(self.log_dir, 'val.pth'))

                        # visualize validation predictions
                        fig = self.vis_pred(self.val_loader)
                        self.writer.add_figure('val/prediction', fig, e)

    def vis_pred(self, loader):
        fig = plt.figure()
        return fig


class TrainerLSS(TrainerCore):
    def __init__(self, dphys_cfg, lss_cfg, model='lss', bsz=1, lr=1e-3, weight_decay=1e-7, nepochs=1000,
                 pretrained_model_path=None, debug=False, vis=False, terrain_weight=1.0, phys_weight=0.1):
        super().__init__(dphys_cfg, lss_cfg, model, bsz, lr, weight_decay, nepochs, pretrained_model_path, debug, vis,
                         terrain_weight, phys_weight)
        # create dataloaders
        self.train_loader, self.val_loader = self.create_dataloaders(bsz=bsz, debug=debug, vis=vis, Data=ROUGH)

        # load models: terrain encoder
        self.terrain_encoder = load_lss_model(modelf=pretrained_model_path, lss_cfg=self.lss_cfg, device=self.device)
        self.terrain_encoder.train()

        # define optimizer
        self.optimizer = torch.optim.Adam(self.terrain_encoder.parameters(), lr=lr, weight_decay=weight_decay)

    def compute_losses(self, batch):
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_terrain,
         control_ts, controls,
         traj_ts, Xs, Xds, Rs, Omegas) = batch

        terrain, weights_terrain = hm_terrain[:, 0:1], hm_terrain[:, 1:2]

        # terrain encoder forward pass
        inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        out = self.terrain_encoder(*inputs)
        terrain_pred, friction_pred = out['terrain'], out['friction']

        # rigid / terrain height map loss
        if self.terrain_weight > 0:
            loss_terrain = self.terrain_hm_loss(terrain_pred, terrain, weights_terrain)
        else:
            loss_terrain = torch.tensor(0.0, device=self.device)

        # physics loss: difference between predicted and ground truth states
        states_gt = [Xs, Xds, Rs, Omegas]
        states_pred, _ = self.dphysics(z_grid=terrain_pred.squeeze(1), controls=controls,
                                       friction=friction_pred.squeeze(1))
        if self.phys_weight > 0:
            loss_phys = self.physics_loss(states_pred=states_pred, states_gt=states_gt,
                                          pred_ts=control_ts, gt_ts=traj_ts)
        else:
            loss_phys = torch.tensor(0.0, device=self.device)

        return loss_terrain, loss_phys

    def vis_pred(self, loader):
        fig = plt.figure(figsize=(16, 4))
        ax1 = fig.add_subplot(241)
        ax2 = fig.add_subplot(242)
        ax3 = fig.add_subplot(243)
        ax4 = fig.add_subplot(244)
        ax5 = fig.add_subplot(245)
        ax6 = fig.add_subplot(246)
        ax7 = fig.add_subplot(247)
        ax8 = fig.add_subplot(248)

        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
        for ax in axes:
            ax.clear()

        # visualize training predictions
        sample_i = np.random.choice(len(loader.dataset))
        sample = loader.dataset[sample_i]

        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_terrain,
         controls_ts, controls,
         traj_ts, Xs, Xds, Rs, Omegas) = sample
        with torch.no_grad():
            # predict height maps
            img_inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
            img_inputs = [i.unsqueeze(0).to(self.device) for i in img_inputs]  # add batch dimension and move to device
            out = self.terrain_encoder(*img_inputs)
            terrain_pred, friction_pred = out['terrain'], out['friction']

            # predict states
            states_pred, _ = self.dphysics(z_grid=terrain_pred.squeeze(1),
                                           controls=controls.unsqueeze(0).to(self.device),
                                           friction=friction_pred.squeeze(1))
        terrain_pred = terrain_pred[0, 0].cpu()
        friction_pred = friction_pred[0, 0].cpu()
        Xs_pred = states_pred[0][0].cpu()

        # get height map points
        z_grid = terrain_pred
        x_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
        y_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
        x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
        hm_points = torch.stack([x_grid, y_grid, z_grid], dim=-1)
        hm_points = hm_points.view(-1, 3).T

        # plot images with projected height map points
        for imgi in range(imgs.shape[0])[:4]:
            ax = axes[imgi]
            img = imgs[imgi]
            img = denormalize_img(img[:3])
            cam_pts = ego_to_cam(hm_points, rots[imgi], trans[imgi], intrins[imgi])
            img_H, img_W = self.lss_cfg['data_aug_conf']['H'], self.lss_cfg['data_aug_conf']['W']
            mask_img = get_only_in_img_mask(cam_pts, img_H, img_W)
            plot_pts = post_rots[imgi].matmul(cam_pts) + post_trans[imgi].unsqueeze(1)
            ax.imshow(img)
            ax.scatter(plot_pts[0, mask_img], plot_pts[1, mask_img], s=1, c=hm_points[2, mask_img],
                       cmap='jet', vmin=-1.0, vmax=1.0)
            ax.axis('off')

        ax5.set_title('Prediction: Terrain')
        ax5.imshow(terrain_pred.T, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)

        ax6.set_title('Label: Terrain')
        ax6.imshow(hm_terrain[0].T, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)

        ax7.set_title('Friction')
        ax7.imshow(friction_pred.T, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)

        ax8.set_title('Trajectories')
        ax8.plot(Xs[:, 0], Xs[:, 1], 'kx', label='GT')
        ax8.plot(Xs_pred[:, 0], Xs_pred[:, 1], 'r.', label='Pred')
        ax8.set_xlabel('X [m]')
        ax8.set_ylabel('Y [m]')
        ax8.grid()
        ax8.legend()
        ax8.axis('equal')

        return fig


class Fusion(ROUGH):
    def __init__(self, path, lss_cfg=None, dphys_cfg=DPhysConfig(), is_train=True):
        super(Fusion, self).__init__(path, lss_cfg, dphys_cfg=dphys_cfg, is_train=is_train)

    def get_cloud(self, i, points_source='lidar'):
        cloud = self.get_raw_cloud(i)
        # move points to robot frame
        Tr = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr = np.asarray(Tr, dtype=float).reshape((4, 4))
        cloud = transform_cloud(cloud, Tr)
        return cloud

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_images_data(i)
        points = torch.as_tensor(position(self.get_cloud(i))).T
        control_ts, controls = self.get_controls(i)
        traj_ts, states = self.get_states_traj(i)
        Xs, Xds, Rs, Omegas = states
        hm_terrain = self.get_terrain_height_map(i)

        return (imgs, rots, trans, intrins, post_rots, post_trans,
                hm_terrain,
                control_ts, controls,
                traj_ts, Xs, Xds, Rs, Omegas,
                points)

class TrainerBEVFusion(TrainerCore):

    def __init__(self, dphys_cfg, lss_cfg, model='bevfusion', bsz=1, lr=1e-3, weight_decay=1e-7, nepochs=1000,
                 pretrained_model_path=None, debug=False, vis=False, terrain_weight=1.0, phys_weight=0.1):
        super().__init__(dphys_cfg, lss_cfg, model, bsz, lr, weight_decay, nepochs, pretrained_model_path, debug, vis,
                         terrain_weight, phys_weight)
        # create dataloaders
        self.train_loader, self.val_loader = self.create_dataloaders(bsz=bsz, debug=debug, vis=vis, Data=Fusion)

        # load models: terrain encoder
        self.terrain_encoder = BEVFusion(grid_conf=self.lss_cfg['grid_conf'],
                                         data_aug_conf=self.lss_cfg['data_aug_conf']).to(self.device)
        self.terrain_encoder.train()

        # define optimizer
        self.optimizer = torch.optim.Adam(self.terrain_encoder.parameters(), lr=lr, weight_decay=weight_decay)

    def compute_losses(self, batch):
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_terrain,
         control_ts, controls,
         traj_ts, Xs, Xds, Rs, Omegas,
         points) = batch

        terrain, weights_terrain = hm_terrain[:, 0:1], hm_terrain[:, 1:2]

        # terrain encoder forward pass
        img_inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        points_input = points
        out = self.terrain_encoder(img_inputs, points_input)
        terrain_pred, friction_pred = out['terrain'], out['friction']

        # rigid / terrain height map loss
        if self.terrain_weight > 0:
            loss_terrain = self.terrain_hm_loss(terrain_pred, terrain, weights_terrain)
        else:
            loss_terrain = torch.tensor(0.0, device=self.device)

        # physics loss: difference between predicted and ground truth states
        states_gt = [Xs, Xds, Rs, Omegas]
        states_pred, _ = self.dphysics(z_grid=terrain_pred.squeeze(1), controls=controls,
                                       friction=friction_pred.squeeze(1))
        if self.phys_weight > 0:
            loss_phys = self.physics_loss(states_pred=states_pred, states_gt=states_gt,
                                          pred_ts=control_ts, gt_ts=traj_ts)
        else:
            loss_phys = torch.tensor(0.0, device=self.device)

        return loss_terrain, loss_phys

    def vis_pred(self, loader):
        fig = plt.figure(figsize=(16, 4))
        ax1 = fig.add_subplot(241)
        ax2 = fig.add_subplot(242)
        ax3 = fig.add_subplot(243)
        ax4 = fig.add_subplot(244)
        ax5 = fig.add_subplot(245)
        ax6 = fig.add_subplot(246)
        ax7 = fig.add_subplot(247)
        ax8 = fig.add_subplot(248)

        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
        for ax in axes:
            ax.clear()

        # visualize training predictions
        sample_i = np.random.choice(len(loader.dataset))
        sample = loader.dataset[sample_i]

        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_terrain,
         controls_ts, controls,
         traj_ts, Xs, Xds, Rs, Omegas,
         points) = sample
        with torch.no_grad():
            # predict height maps
            img_inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
            img_inputs = [i.unsqueeze(0).to(self.device) for i in img_inputs]  # add batch dimension and move to device
            points_input = points.unsqueeze(0).to(self.device)
            out = self.terrain_encoder(img_inputs, points_input)
            terrain_pred, friction_pred = out['terrain'], out['friction']

            # predict states
            states_pred, _ = self.dphysics(z_grid=terrain_pred.squeeze(1),
                                           controls=controls.unsqueeze(0).to(self.device),
                                           friction=friction_pred.squeeze(1))
        terrain_pred = terrain_pred[0, 0].cpu()
        friction_pred = friction_pred[0, 0].cpu()
        Xs_pred = states_pred[0][0].cpu()

        # get height map points
        z_grid = terrain_pred
        x_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
        y_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
        x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
        hm_points = torch.stack([x_grid, y_grid, z_grid], dim=-1)
        hm_points = hm_points.view(-1, 3).T

        # plot images with projected height map points
        for imgi in range(imgs.shape[0])[:4]:
            ax = axes[imgi]
            img = imgs[imgi]
            img = denormalize_img(img[:3])
            cam_pts = ego_to_cam(hm_points, rots[imgi], trans[imgi], intrins[imgi])
            img_H, img_W = self.lss_cfg['data_aug_conf']['H'], self.lss_cfg['data_aug_conf']['W']
            mask_img = get_only_in_img_mask(cam_pts, img_H, img_W)
            plot_pts = post_rots[imgi].matmul(cam_pts) + post_trans[imgi].unsqueeze(1)
            ax.imshow(img)
            ax.scatter(plot_pts[0, mask_img], plot_pts[1, mask_img], s=1, c=hm_points[2, mask_img],
                       cmap='jet', vmin=-1.0, vmax=1.0)
            ax.axis('off')

        ax5.set_title('Prediction: Terrain')
        ax5.imshow(terrain_pred.T, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)

        ax6.set_title('Label: Terrain')
        ax6.imshow(hm_terrain[0].T, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)

        ax7.set_title('Friction')
        ax7.imshow(friction_pred.T, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)

        ax8.set_title('Trajectories')
        ax8.plot(Xs[:, 0], Xs[:, 1], 'kx', label='GT')
        ax8.plot(Xs_pred[:, 0], Xs_pred[:, 1], 'r.', label='Pred')
        ax8.set_xlabel('X [m]')
        ax8.set_ylabel('Y [m]')
        ax8.grid()
        ax8.legend()
        ax8.axis('equal')

        return fig


class Points(ROUGH):
    def __init__(self, path, lss_cfg=None, dphys_cfg=DPhysConfig(), is_train=True):
        super(Points, self).__init__(path, lss_cfg, dphys_cfg=dphys_cfg, is_train=is_train)

    def get_cloud(self, i, points_source='lidar'):
        cloud = self.get_raw_cloud(i)
        # move points to robot frame
        Tr = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr = np.asarray(Tr, dtype=float).reshape((4, 4))
        cloud = transform_cloud(cloud, Tr)
        return cloud

    def get_sample(self, i):
        points = torch.as_tensor(position(self.get_cloud(i))).T
        control_ts, controls = self.get_controls(i)
        traj_ts, states = self.get_states_traj(i)
        Xs, Xds, Rs, Omegas = states
        hm_terrain = self.get_terrain_height_map(i)
        return (points, hm_terrain,
                control_ts, controls,
                traj_ts, Xs, Xds, Rs, Omegas)

class TrainerLiDARBEV(TrainerCore):
        def __init__(self, dphys_cfg, lss_cfg, model='lidarbev', bsz=1, lr=1e-3, weight_decay=1e-7, nepochs=1000,
                    pretrained_model_path=None, debug=False, vis=False, terrain_weight=1.0, phys_weight=0.1):
            super().__init__(dphys_cfg, lss_cfg, model, bsz, lr, weight_decay, nepochs, pretrained_model_path, debug, vis,
                            terrain_weight, phys_weight)
            # create dataloaders
            self.train_loader, self.val_loader = self.create_dataloaders(bsz=bsz, debug=debug, vis=vis, Data=Points)

            # load models: terrain encoder
            self.terrain_encoder = LidarBEV(grid_conf=self.lss_cfg['grid_conf'], outC=1).to(self.device)
            self.terrain_encoder.train()

            # define optimizer
            self.optimizer = torch.optim.Adam(self.terrain_encoder.parameters(), lr=lr, weight_decay=weight_decay)

        def compute_losses(self, batch):
            (points, hm_terrain,
             control_ts, controls,
             traj_ts, Xs, Xds, Rs, Omegas) = batch

            terrain, weights_terrain = hm_terrain[:, 0:1], hm_terrain[:, 1:2]

            # terrain encoder forward pass
            points_input = points
            out = self.terrain_encoder(points_input)
            terrain_pred = out['terrain']
            friction_pred = out['friction']

            # rigid / terrain height map loss
            if self.terrain_weight > 0:
                loss_terrain = self.terrain_hm_loss(terrain_pred, terrain, weights_terrain)
            else:
                loss_terrain = torch.tensor(0.0, device=self.device)

            # physics loss: difference between predicted and ground truth states
            states_gt = [Xs, Xds, Rs, Omegas]
            states_pred, _ = self.dphysics(z_grid=terrain_pred.squeeze(1), controls=controls, friction=friction_pred.squeeze(1))
            if self.phys_weight > 0:
                loss_phys = self.physics_loss(states_pred=states_pred, states_gt=states_gt,
                                              pred_ts=control_ts, gt_ts=traj_ts)
            else:
                loss_phys = torch.tensor(0.0, device=self.device)

            return loss_terrain, loss_phys

        def vis_pred(self, loader):
            sample_i = np.random.choice(len(loader.dataset))
            sample = loader.dataset[sample_i]
            (points, hm_terrain,
             control_ts, controls,
             traj_ts, Xs, Xds, Rs, Omegas) = sample
            terrain, weights = hm_terrain[0], hm_terrain[1]

            # predict terrain and friction
            out = self.terrain_encoder(points.unsqueeze(0).to(self.device))  # (B, outC, H, W)
            terrain_pred = out['terrain']
            friction_pred = out['friction']

            # predict states
            states_pred, _ = self.dphysics(z_grid=terrain_pred.squeeze(1),
                                           controls=controls.unsqueeze(0).to(self.device),
                                           friction=friction_pred.squeeze(1))

            terrain_pred = terrain_pred[0, 0].cpu()
            friction_pred = friction_pred[0, 0].cpu()
            Xs_pred = states_pred[0][0].cpu()

            # visualize the output
            fig, ax = plt.subplots(1, 5, figsize=(20, 4))

            ax[0].imshow(terrain_pred.T, cmap='jet', origin='lower', vmin=-1, vmax=1)
            ax[0].set_title('Predicted Terrain')

            ax[1].imshow(terrain.T, cmap='jet', origin='lower', vmin=-1, vmax=1)
            ax[1].set_title('Ground truth')

            ax[2].imshow(weights.T, cmap='gray', origin='lower')
            ax[2].set_title('Weights')

            ax[3].imshow(friction_pred.T, cmap='jet', origin='lower', vmin=0, vmax=1)
            ax[3].set_title('Friction')

            ax[4].plot(Xs[:, 0], Xs[:, 1], 'kx', label='GT')
            ax[4].plot(Xs_pred[:, 0], Xs_pred[:, 1], 'r.', label='Pred')
            ax[4].set_title('Trajectories')
            ax[4].set_xlabel('X [m]')
            ax[4].set_ylabel('Y [m]')
            ax[4].legend()
            ax[4].grid()
            ax[4].axis('equal')

            return fig


def choose_trainer(model):
    if model == 'lss':
        return TrainerLSS
    elif model == 'bevfusion':
        return TrainerBEVFusion
    elif model == 'lidarbev':
        return TrainerLiDARBEV
    else:
        raise ValueError(f'Invalid model: {model}. Supported models: lss, bevfusion, lidarbev')


def main():
    args = arg_parser()
    print(args)

    # load configs: DPhys and LSS (terrain encoder)
    dphys_cfg = DPhysConfig(robot=args.robot)
    lss_config_path = args.lss_cfg_path
    assert os.path.isfile(lss_config_path), 'LSS config file %s does not exist' % lss_config_path
    lss_cfg = read_yaml(lss_config_path)

    # create trainer
    Trainer = choose_trainer(args.model)
    trainer = Trainer(model=args.model,
                      dphys_cfg=dphys_cfg, lss_cfg=lss_cfg,
                      bsz=args.bsz, nepochs=args.nepochs,
                      lr=args.lr, weight_decay=args.weight_decay,
                      pretrained_model_path=args.pretrained_model_path,
                      terrain_weight=args.terrain_weight,
                      phys_weight=args.phys_weight,
                      debug=args.debug, vis=args.vis)
    # start training
    trainer.train()


if __name__ == '__main__':
    main()
