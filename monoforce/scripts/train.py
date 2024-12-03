#!/usr/bin/env python

import sys
sys.path.append('../src')
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from monoforce.models.terrain_encoder.utils import denormalize_img, ego_to_cam, get_only_in_img_mask
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.terrain_encoder.bevfusion import BEVFusion
from monoforce.models.terrain_encoder.lidarbev import LidarBEV
from monoforce.models.dphysics import DPhysics
from monoforce.dphys_config import DPhysConfig
from monoforce.datasets.rough import ROUGH
from monoforce.utils import read_yaml, write_to_yaml, str2bool, compile_data, position
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
    parser.add_argument('--bsz', type=int, default=1, help='Batch size')
    parser.add_argument('--nepochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay')
    parser.add_argument('--robot', type=str, default='marv', help='Robot name')
    parser.add_argument('--lss_cfg_path', type=str, default='../config/lss_cfg.yaml', help='Path to LSS config')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--debug', type=str2bool, default=True, help='Debug mode: use small datasets')
    parser.add_argument('--vis', type=str2bool, default=False, help='Visualize training samples')
    parser.add_argument('--geom_weight', type=float, default=1.0, help='Weight for geometry loss')
    parser.add_argument('--terrain_weight', type=float, default=2.0, help='Weight for terrain heightmap loss')
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
                 geom_weight=1.0,
                 terrain_weight=1.0,
                 phys_weight=0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'rough'
        self.model = model
        self.dphys_cfg = dphys_cfg
        self.lss_cfg = lss_cfg
        self.debug = debug

        self.nepochs = nepochs
        self.min_loss = np.inf
        self.min_train_loss = np.inf
        self.train_counter = 0
        self.val_counter = 0

        self.geom_weight = geom_weight
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
        train_ds, val_ds = compile_data(small_data=debug, vis=vis, Data=Data)
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

    def hm_loss(self, height_pred, height_gt, weights=None):
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

        return loss

    def physics_loss(self, states_pred, states_gt, pred_ts, gt_ts, only_2d=False):
        # unpack states
        X, Xd, R, Omega = states_gt
        X_pred, Xd_pred, R_pred, Omega_pred = states_pred

        # find the closest timesteps in the trajectory to the ground truth timesteps
        ts_ids = torch.argmin(torch.abs(pred_ts.unsqueeze(1) - gt_ts.unsqueeze(2)), dim=2)

        # get the predicted states at the closest timesteps to the ground truth timesteps
        batch_size = X.shape[0]
        X_pred_gt_ts = X_pred[torch.arange(batch_size).unsqueeze(1), ts_ids]

        # trajectory loss
        loss_fn = torch.nn.functional.mse_loss
        if only_2d:
            loss = loss_fn(X_pred_gt_ts[..., :2], X[..., :2])
        else:
            loss = loss_fn(X_pred_gt_ts, X)

        return loss

    def compute_losses(self, batch):
        loss_geom = torch.tensor(0.0, device=self.device)
        loss_terrain = torch.tensor(0.0, device=self.device)
        loss_phys = torch.tensor(0.0, device=self.device)
        return loss_geom, loss_terrain, loss_phys

    def epoch(self, train=True):
        loader = self.train_loader if train else self.val_loader
        counter = self.train_counter if train else self.val_counter

        if train:
            self.terrain_encoder.train()
        else:
            self.terrain_encoder.eval()

        max_grad_norm = 5.0
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
            epoch_losses['total'] += loss.item()

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
        # save configs to log dir
        write_to_yaml(self.dphys_cfg.__dict__, os.path.join(self.log_dir, 'dphys_cfg.yaml'))
        write_to_yaml(self.lss_cfg, os.path.join(self.log_dir, 'lss_cfg.yaml'))

        for e in range(self.nepochs):
            # training epoch
            train_losses, self.train_counter = self.epoch(train=True)
            for k, v in train_losses.items():
                print('Epoch:', e, f'Train loss {k}:', v)
                self.writer.add_scalar(f'train/epoch_loss_{k}', v, e)

            if train_losses['total'] < self.min_train_loss or self.debug:
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

                    if val_losses['total'] < self.min_loss or self.debug:
                        self.min_loss = val_losses['total']
                        print('Saving model...')
                        self.terrain_encoder.eval()
                        torch.save(self.terrain_encoder.state_dict(), os.path.join(self.log_dir, 'val.pth'))

                        # visualize validation predictions
                        fig = self.vis_pred(self.val_loader)
                        self.writer.add_figure('val/prediction', fig, e)

    def pred(self, sample):
        raise NotImplementedError

    def predicts_states(self, terrain, pose0, controls):
        x0 = pose0[:, :3, 3].to(self.device)
        xd0 = torch.zeros_like(x0)
        R0 = pose0[:, :3, :3].to(self.device)
        omega0 = torch.zeros_like(xd0)
        state0 = (x0, xd0, R0, omega0)
        states_pred, _ = self.dphysics(z_grid=terrain['terrain'].squeeze(1), state=state0,
                                       controls=controls.to(self.device),
                                       friction=terrain['friction'].squeeze(1))
        return states_pred

    def vis_pred(self, loader):
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))

        # visualize training predictions
        sample_i = np.random.choice(len(loader.dataset))
        sample = loader.dataset[sample_i]

        if self.model == 'lss':
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_geom, hm_terrain,
             controls_ts, controls,
             pose0,
             traj_ts, Xs, Xds, Rs, Omegas) = sample
        elif self.model == 'bevfusion':
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_geom, hm_terrain,
             controls_ts, controls,
             pose0,
             traj_ts, Xs, Xds, Rs, Omegas,
             points) = sample
        elif self.model == 'lidarbev':
            (points, hm_geom, hm_terrain,
             controls_ts, controls,
             pose0,
             traj_ts, Xs, Xds, Rs, Omegas) = sample
        else:
            raise ValueError('Model not supported')

        # predict height maps and states
        with torch.no_grad():
            batch = [torch.as_tensor(b, dtype=torch.float32, device=self.device).unsqueeze(0) for b in sample]
            terrain, states_pred = self.pred(batch)

        geom_pred = terrain['geom'][0, 0].cpu()
        diff_pred = terrain['diff'][0, 0].cpu()
        terrain_pred = terrain['terrain'][0, 0].cpu()
        friction_pred = terrain['friction'][0, 0].cpu()
        Xs_pred = states_pred[0][0].cpu()

        # get height map points
        z_grid = terrain_pred
        x_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
        y_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
        x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
        hm_points = torch.stack([x_grid, y_grid, z_grid], dim=-1)
        hm_points = hm_points.view(-1, 3).T

        if self.model in ['lss', 'bevfusion']:
            # plot images with projected height map points
            for imgi in range(imgs.shape[0])[:4]:
                ax = axes[0, imgi]
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

        axes[1, 0].set_title('Prediction: Terrain')
        axes[1, 0].imshow(terrain_pred.T, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)

        axes[1, 1].set_title('Label: Terrain')
        axes[1, 1].imshow(hm_terrain[0].T, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)

        axes[1, 2].set_title('Friction')
        axes[1, 2].imshow(friction_pred.T, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)

        axes[1, 3].set_title('Trajectories XY')
        axes[1, 3].plot(Xs[:, 0], Xs[:, 1], 'kx', label='GT')
        axes[1, 3].plot(Xs_pred[:, 0], Xs_pred[:, 1], 'r.', label='Pred')
        axes[1, 3].set_xlabel('X [m]')
        axes[1, 3].set_ylabel('Y [m]')
        axes[1, 3].set_xlim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
        axes[1, 3].set_ylim(-self.dphys_cfg.d_max, self.dphys_cfg.d_max)
        axes[1, 3].grid()
        axes[1, 3].legend()

        axes[2, 0].set_title('Prediction: Geom')
        axes[2, 0].imshow(geom_pred.T, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)

        axes[2, 1].set_title('Label: Geom')
        axes[2, 1].imshow(hm_geom[0].T, origin='lower', cmap='jet', vmin=-1.0, vmax=1.0)

        axes[2, 2].set_title('Height diff')
        axes[2, 2].imshow(diff_pred.T, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)

        axes[2, 3].set_title('Trajectories Z')
        axes[2, 3].plot(traj_ts, Xs[:, 2], 'kx', label='GT')
        axes[2, 3].plot(controls_ts, Xs_pred[:, 2], 'r.', label='Pred')
        axes[2, 3].set_xlabel('Time [s]')
        axes[2, 3].set_ylabel('Z [m]')
        axes[2, 3].set_ylim(-self.dphys_cfg.h_max, self.dphys_cfg.h_max)
        axes[2, 3].grid()
        axes[2, 3].legend()

        return fig


class TrainerLSS(TrainerCore):
    def __init__(self, dphys_cfg, lss_cfg, model='lss', bsz=1, lr=1e-3, weight_decay=1e-7, nepochs=1000,
                 pretrained_model_path=None, debug=False, vis=False, geom_weight=1.0, terrain_weight=1.0, phys_weight=0.1):
        super().__init__(dphys_cfg, lss_cfg, model, bsz, lr, weight_decay, nepochs, pretrained_model_path, debug, vis,
                         geom_weight, terrain_weight, phys_weight)
        # create dataloaders
        self.train_loader, self.val_loader = self.create_dataloaders(bsz=bsz, debug=debug, vis=vis, Data=ROUGH)

        # load models: terrain encoder
        self.terrain_encoder = LiftSplatShoot(self.lss_cfg['grid_conf'],
                                              self.lss_cfg['data_aug_conf']).from_pretrained(pretrained_model_path)
        self.terrain_encoder.to(self.device)
        self.terrain_encoder.train()

        # define optimizer
        self.optimizer = torch.optim.Adam(self.terrain_encoder.parameters(), lr=lr, weight_decay=weight_decay)

    def compute_losses(self, batch):
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain,
         control_ts, controls,
         pose0,
         traj_ts, Xs, Xds, Rs, Omegas) = batch
        # terrain encoder forward pass
        inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        terrain = self.terrain_encoder(*inputs)

        # geometry loss: difference between predicted and ground truth height maps
        if self.geom_weight > 0:
            loss_geom = self.hm_loss(terrain['geom'], hm_geom[:, 0:1], hm_geom[:, 1:2])
        else:
            loss_geom = torch.tensor(0.0, device=self.device)

        # rigid / terrain height map loss
        if self.terrain_weight > 0:
            loss_terrain = self.hm_loss(terrain['terrain'], hm_terrain[:, 0:1], hm_terrain[:, 1:2])
        else:
            loss_terrain = torch.tensor(0.0, device=self.device)

        # physics loss: difference between predicted and ground truth states
        states_gt = [Xs, Xds, Rs, Omegas]
        states_pred = self.predicts_states(terrain, pose0, controls)

        if self.phys_weight > 0:
            loss_phys = self.physics_loss(states_pred=states_pred, states_gt=states_gt,
                                          pred_ts=control_ts, gt_ts=traj_ts)
        else:
            loss_phys = torch.tensor(0.0, device=self.device)

        return loss_geom, loss_terrain, loss_phys

    def pred(self, batch):
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain,
         controls_ts, controls,
         pose0,
         traj_ts, Xs, Xds, Rs, Omegas) = batch
        # predict height maps
        img_inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        terrain = self.terrain_encoder(*img_inputs)

        # predict states
        states_pred = self.predicts_states(terrain, pose0, controls)

        return terrain, states_pred


class Fusion(ROUGH):
    def __init__(self, path, lss_cfg=None, dphys_cfg=DPhysConfig(), is_train=True):
        super(Fusion, self).__init__(path, lss_cfg, dphys_cfg=dphys_cfg, is_train=is_train)

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

class TrainerBEVFusion(TrainerCore):

    def __init__(self, dphys_cfg, lss_cfg, model='bevfusion', bsz=1, lr=1e-3, weight_decay=1e-7, nepochs=1000,
                 pretrained_model_path=None, debug=False, vis=False, geom_weight=1.0, terrain_weight=1.0, phys_weight=0.1):
        super().__init__(dphys_cfg, lss_cfg, model, bsz, lr, weight_decay, nepochs, pretrained_model_path, debug, vis,
                         geom_weight, terrain_weight, phys_weight)
        # create dataloaders
        self.train_loader, self.val_loader = self.create_dataloaders(bsz=bsz, debug=debug, vis=vis, Data=Fusion)

        # load models: terrain encoder
        self.terrain_encoder = BEVFusion(grid_conf=self.lss_cfg['grid_conf'],
                                         data_aug_conf=self.lss_cfg['data_aug_conf']).from_pretrained(pretrained_model_path)
        self.terrain_encoder.to(self.device)
        self.terrain_encoder.train()

        # define optimizer
        self.optimizer = torch.optim.Adam(self.terrain_encoder.parameters(), lr=lr, weight_decay=weight_decay)

    def compute_losses(self, batch):
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain,
         control_ts, controls,
         pose0,
         traj_ts, Xs, Xds, Rs, Omegas,
         points) = batch
        # terrain encoder forward pass
        img_inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        points_input = points
        terrain = self.terrain_encoder(img_inputs, points_input)

        # geometry loss: difference between predicted and ground truth height maps
        if self.geom_weight > 0:
            loss_geom = self.hm_loss(terrain['geom'], hm_geom[:, 0:1], hm_geom[:, 1:2])
        else:
            loss_geom = torch.tensor(0.0, device=self.device)

        # rigid / terrain height map loss
        if self.terrain_weight > 0:
            loss_terrain = self.hm_loss(terrain['terrain'], hm_terrain[:, 0:1], hm_terrain[:, 1:2])
        else:
            loss_terrain = torch.tensor(0.0, device=self.device)

        # physics loss: difference between predicted and ground truth states
        states_gt = [Xs, Xds, Rs, Omegas]
        states_pred = self.predicts_states(terrain, pose0, controls)

        if self.phys_weight > 0:
            loss_phys = self.physics_loss(states_pred=states_pred, states_gt=states_gt,
                                          pred_ts=control_ts, gt_ts=traj_ts)
        else:
            loss_phys = torch.tensor(0.0, device=self.device)

        return loss_geom, loss_terrain, loss_phys

    def pred(self, batch):
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain,
         controls_ts, controls,
         pose0,
         traj_ts, Xs, Xds, Rs, Omegas,
         points) = batch

        # predict height maps
        img_inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        terrain = self.terrain_encoder(img_inputs, points)

        # predict states
        states_pred = self.predicts_states(terrain, pose0, controls)

        return terrain, states_pred


class Points(ROUGH):
    def __init__(self, path, lss_cfg=None, dphys_cfg=DPhysConfig(), is_train=True):
        super(Points, self).__init__(path, lss_cfg, dphys_cfg=dphys_cfg, is_train=is_train)

    def get_sample(self, i):
        points = torch.as_tensor(position(self.get_cloud(i))).T
        control_ts, controls = self.get_controls(i)
        traj_ts, states = self.get_states_traj(i)
        Xs, Xds, Rs, Omegas = states
        hm_geom = self.get_geom_height_map(i)
        hm_terrain = self.get_terrain_height_map(i)
        pose0 = torch.as_tensor(self.get_initial_pose_on_heightmap(i), dtype=torch.float32)
        return (points, hm_geom, hm_terrain,
                control_ts, controls,
                pose0,
                traj_ts, Xs, Xds, Rs, Omegas)

class TrainerLidarBEV(TrainerCore):
        def __init__(self, dphys_cfg, lss_cfg, model='lidarbev', bsz=1, lr=1e-3, weight_decay=1e-7, nepochs=1000,
                    pretrained_model_path=None, debug=False, vis=False, geom_weight=1.0, terrain_weight=1.0, phys_weight=0.1):
            super().__init__(dphys_cfg, lss_cfg, model, bsz, lr, weight_decay, nepochs, pretrained_model_path, debug, vis,
                             geom_weight, terrain_weight, phys_weight)
            # create dataloaders
            self.train_loader, self.val_loader = self.create_dataloaders(bsz=bsz, debug=debug, vis=vis, Data=Points)

            # load models: terrain encoder
            self.terrain_encoder = LidarBEV(grid_conf=self.lss_cfg['grid_conf'],
                                            outC=1).from_pretrained(pretrained_model_path)
            self.terrain_encoder.to(self.device)
            self.terrain_encoder.train()

            # define optimizer
            self.optimizer = torch.optim.Adam(self.terrain_encoder.parameters(), lr=lr, weight_decay=weight_decay)

        def compute_losses(self, batch):
            (points, hm_geom, hm_terrain,
             control_ts, controls,
             pose0,
             traj_ts, Xs, Xds, Rs, Omegas) = batch
            # terrain encoder forward pass
            points_input = points
            terrain = self.terrain_encoder(points_input)

            # geometry loss: difference between predicted and ground truth height maps
            if self.geom_weight > 0:
                loss_geom = self.hm_loss(terrain['geom'], hm_geom[:, 0:1], hm_geom[:, 1:2])
            else:
                loss_geom = torch.tensor(0.0, device=self.device)

            # rigid / terrain height map loss
            if self.terrain_weight > 0:
                loss_terrain = self.hm_loss(terrain['terrain'], hm_terrain[:, 0:1], hm_terrain[:, 1:2])
            else:
                loss_terrain = torch.tensor(0.0, device=self.device)

            # physics loss: difference between predicted and ground truth states
            states_gt = [Xs, Xds, Rs, Omegas]
            states_pred = self.predicts_states(terrain, pose0, controls)

            if self.phys_weight > 0:
                loss_phys = self.physics_loss(states_pred=states_pred, states_gt=states_gt,
                                              pred_ts=control_ts, gt_ts=traj_ts)
            else:
                loss_phys = torch.tensor(0.0, device=self.device)

            return loss_geom, loss_terrain, loss_phys

        def pred(self, batch):
            (points, hm_geom, hm_terrain,
             control_ts, controls,
             pose0,
             traj_ts, Xs, Xds, Rs, Omegas) = batch

            # predict terrain
            terrain = self.terrain_encoder(points)

            # predict states
            states_pred = self.predicts_states(terrain, pose0, controls)

            return terrain, states_pred

def choose_trainer(model):
    if model == 'lss':
        return TrainerLSS
    elif model == 'bevfusion':
        return TrainerBEVFusion
    elif model == 'lidarbev':
        return TrainerLidarBEV
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
                      geom_weight=args.geom_weight,
                      terrain_weight=args.terrain_weight,
                      phys_weight=args.phys_weight,
                      debug=args.debug, vis=args.vis)
    # start training
    trainer.train()


if __name__ == '__main__':
    main()
