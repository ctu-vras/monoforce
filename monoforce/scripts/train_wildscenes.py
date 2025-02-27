#!/usr/bin/env python

import sys
sys.path.append('../src')
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from monoforce.models.terrain_encoder.utils import denormalize_img, ego_to_cam, get_only_in_img_mask
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.traj_predictor.dphysics import DPhysics
from monoforce.models.traj_predictor.dphys_config import DPhysConfig
from monoforce.datasets.wildscenes import WildScenes
from monoforce.utils import read_yaml, write_to_yaml, str2bool
from monoforce.losses import hm_loss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import argparse


np.random.seed(42)
torch.manual_seed(42)


def arg_parser():
    parser = argparse.ArgumentParser(description='Train MonoForce model')
    parser.add_argument('--model', type=str, default='lss', help='Model to train: lss, bevfusion, voxelnet')
    parser.add_argument('--bsz', type=int, default=8, help='Batch size')
    parser.add_argument('--nepochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay')
    parser.add_argument('--robot', type=str, default='marv', help='Robot name')
    parser.add_argument('--lss_cfg_path', type=str, default='../config/lss_cfg_wildscenes.yaml', help='Path to LSS config')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--debug', type=str2bool, default=True, help='Debug mode: use small datasets')
    parser.add_argument('--vis', type=str2bool, default=False, help='Visualize training samples')
    parser.add_argument('--geom_weight', type=float, default=1.0, help='Weight for geometry loss')
    parser.add_argument('--terrain_weight', type=float, default=2.0, help='Weight for terrain heightmap loss')

    return parser.parse_args()


class TrainerCore:
    """
    Trainer for terrain encoder model

    Args:
    dphys_cfg: DPhysConfig, physical robot-terrain interaction configuration
    lss_cfg: dict, LSS model configuration
    model: str, model to train: lss
    bsz: int, batch size
    lr: float, learning rate
    weight_decay: float, weight decay
    nepochs: int, number of epochs
    pretrained_model_path: str, path to pretrained model
    debug: bool, debug mode: use small datasets
    vis: bool, visualize training samples
    terrain_weight: float, weight for terrain heightmap loss
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
                 terrain_weight=1.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = 'wildscenes'
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

        self.terrain_encoder = None
        self.terrain_encoder_grid_res = self.lss_cfg['grid_conf']['xbound'][2]
        self.dphysics = DPhysics(dphys_cfg, device=self.device)

        # optimizer
        self.optimizer = None
        
        # dataloaders
        self.train_loader = None
        self.val_loader = None

        self.log_dir = os.path.join('../config/tb_runs/',
                                    f'{self.dataset}/{self.model}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def create_dataloaders(self, bsz=1, debug=False):
        # create dataset for LSS model training
        if not debug:
            train_seqs = ['K-01', 'K-03', 'V-03']
            val_seqs = ['V-01', 'V-02']
        else:
            train_seqs = ['V-02']
            val_seqs = ['V-01']
        train_ds = ConcatDataset([WildScenes(seq=seq, is_train=True) for seq in train_seqs])
        val_ds = ConcatDataset([WildScenes(seq=seq, is_train=False) for seq in val_seqs])
        print('Train dataset:', len(train_ds), 'Val dataset:', len(val_ds))

        # create dataloaders: making sure all elements in a batch are tensors
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

    def compute_losses(self, batch):
        loss_geom = torch.tensor(0.0, device=self.device)
        loss_terrain = torch.tensor(0.0, device=self.device)
        return loss_geom, loss_terrain

    def epoch(self, train=True):
        loader = self.train_loader if train else self.val_loader
        counter = self.train_counter if train else self.val_counter

        if train:
            self.terrain_encoder.train()
        else:
            self.terrain_encoder.eval()

        max_grad_norm = 5.0
        epoch_losses = {'geom': 0.0, 'terrain': 0.0, 'total': 0.0}
        for batch in tqdm(loader, total=len(loader)):
            if train:
                self.optimizer.zero_grad()

            batch = [torch.as_tensor(b, dtype=torch.float32, device=self.device) for b in batch]
            loss_geom, loss_terrain = self.compute_losses(batch)
            loss = self.geom_weight * loss_geom + self.terrain_weight * loss_terrain

            if torch.isnan(loss).item():
                torch.save(self.terrain_encoder.state_dict(), os.path.join(self.log_dir, 'train.pth'))
                raise ValueError('Loss is NaN')

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.terrain_encoder.parameters(), max_norm=max_grad_norm)
                self.optimizer.step()

            epoch_losses['geom'] += loss_geom.item()
            epoch_losses['terrain'] += loss_terrain.item()
            epoch_losses['total'] += loss.item()

            counter += 1
            self.writer.add_scalar(f"{'train' if train else 'val'}/iter_loss_geom", loss_geom.item(), counter)
            self.writer.add_scalar(f"{'train' if train else 'val'}/iter_loss_terrain", loss_terrain.item(), counter)
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

    def vis_pred(self, loader):
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))

        # visualize training predictions
        sample_i = np.random.choice(len(loader.dataset))
        sample = loader.dataset[sample_i]
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain) = sample

        # predict height maps and states
        with torch.no_grad():
            batch = [torch.as_tensor(b, dtype=torch.float32, device=self.device).unsqueeze(0) for b in sample]
            terrain = self.pred(batch)

        geom_pred = terrain['geom'][0, 0].cpu()
        diff_pred = terrain['diff'][0, 0].cpu()
        terrain_pred = terrain['terrain'][0, 0].cpu()

        # get height map points
        z_grid = terrain_pred
        x_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.terrain_encoder_grid_res)
        y_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.terrain_encoder_grid_res)
        x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
        hm_points = torch.stack([x_grid, y_grid, z_grid], dim=-1)
        hm_points = hm_points.view(-1, 3).T

        h_min, h_max = -1.0, 1.0
        offset = -1.2
        h_min += offset
        h_max += offset
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
                           cmap='jet', vmin=h_min, vmax=h_max)
                ax.axis('off')

        axes[1, 0].set_title('Prediction: Terrain')
        axes[1, 0].imshow(terrain_pred.T, origin='lower', cmap='jet', vmin=h_min, vmax=h_max)

        axes[1, 1].set_title('Label: Terrain')
        axes[1, 1].imshow(hm_terrain[0].T, origin='lower', cmap='jet', vmin=h_min, vmax=h_max)

        axes[2, 0].set_title('Prediction: Geom')
        axes[2, 0].imshow(geom_pred.T, origin='lower', cmap='jet', vmin=h_min, vmax=h_max)

        axes[2, 1].set_title('Label: Geom')
        axes[2, 1].imshow(hm_geom[0].T, origin='lower', cmap='jet', vmin=h_min, vmax=h_max)

        axes[2, 2].set_title('Height diff')
        axes[2, 2].imshow(diff_pred.T, origin='lower', cmap='jet', vmin=0.0, vmax=1.0)

        return fig


class Trainer(TrainerCore):
    def __init__(self, dphys_cfg, lss_cfg, model='lss', bsz=1, lr=1e-3, weight_decay=1e-7, nepochs=1000,
                 pretrained_model_path=None, debug=False, vis=False, geom_weight=1.0, terrain_weight=1.0):
        super().__init__(dphys_cfg, lss_cfg, model, bsz, lr, weight_decay, nepochs, pretrained_model_path, debug, vis,
                         geom_weight, terrain_weight)
        # create dataloaders
        self.train_loader, self.val_loader = self.create_dataloaders(bsz=bsz, debug=debug)

        # load models: terrain encoder
        self.terrain_encoder = LiftSplatShoot(self.lss_cfg['grid_conf'],
                                              self.lss_cfg['data_aug_conf']).from_pretrained(pretrained_model_path)
        self.terrain_encoder.to(self.device)

        # define optimizer
        self.optimizer = torch.optim.Adam(self.terrain_encoder.parameters(), lr=lr, weight_decay=weight_decay)

    def compute_losses(self, batch):
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain) = batch
        # terrain encoder forward pass
        inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        terrain = self.terrain_encoder(*inputs)

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

        return loss_geom, loss_terrain

    def pred(self, batch):
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain) = batch
        # predict height maps
        img_inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        terrain = self.terrain_encoder(*img_inputs)

        return terrain


def main():
    args = arg_parser()
    print(args)

    # load configs: DPhys and LSS (terrain encoder)
    dphys_cfg = DPhysConfig(robot=args.robot)
    lss_config_path = args.lss_cfg_path
    assert os.path.isfile(lss_config_path), 'LSS config file %s does not exist' % lss_config_path
    lss_cfg = read_yaml(lss_config_path)

    # create trainer
    trainer = Trainer(model=args.model,
                      dphys_cfg=dphys_cfg, lss_cfg=lss_cfg,
                      bsz=args.bsz, nepochs=args.nepochs,
                      lr=args.lr, weight_decay=args.weight_decay,
                      pretrained_model_path=args.pretrained_model_path,
                      geom_weight=args.geom_weight,
                      terrain_weight=args.terrain_weight,
                      debug=args.debug, vis=args.vis)
    # start training
    trainer.train()


if __name__ == '__main__':
    main()
