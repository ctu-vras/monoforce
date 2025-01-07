#!/usr/bin/env python

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from monoforce.models.traj_predictor.traj_lstm import TrajLSTM
from monoforce.datasets.rough import ROUGH
from monoforce.utils import compile_data


class Data(ROUGH):
    def __init__(self, path, is_train=False, **kwargs):
        super(Data, self).__init__(path, is_train=is_train)

    def get_sample(self, i):
        control_ts, controls = self.get_controls(i)

        traj = self.get_traj(i)
        traj_ts = traj['stamps']
        traj_ts = torch.as_tensor(traj_ts - traj_ts[0], dtype=torch.float32)

        poses = traj['poses']
        xyz = torch.as_tensor(poses[:, :3, 3], dtype=torch.float32)
        Rs = poses[:, :3, :3]
        rpy = torch.as_tensor(Rotation.from_matrix(Rs).as_euler('xyz'), dtype=torch.float32)
        states = torch.cat((xyz, rpy), dim=-1)

        hm = self.get_geom_height_map(i)[0:1]
        return (hm,
                control_ts, controls,
                traj_ts, states)


class Trainer:
    def __init__(self,
                 state_features=6,  # (x, y, z, roll, pitch, yaw)
                 control_features=2,  # (linear velocity, angular velocity)
                 heightmap_shape=(128, 128),
                 batch_size=1,
                 lr=1e-4,
                 n_epochs=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Instantiate the model
        self.lstm = TrajLSTM(state_features, control_features, heightmap_shape)
        self.lstm.train()
        self.lstm.to(self.device)

        # Dataset
        self.batch_size = batch_size
        self.train_loader, self.val_loader = self.create_dataloaders()

        # Loss function and optimizer
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=lr)

        self.n_epochs = n_epochs
        self.dataset = 'rough'
        self.model = 'lstm'
        self.log_dir = os.path.join('../config/tb_runs/',
                                    f'{self.dataset}/{self.model}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def create_dataloaders(self):
        train_ds, val_ds = compile_data(Data=Data, small_data=False)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def inference(self, batch):
        hm, control_ts, controls, traj_ts, xyz_rpy = batch
        xyz_rpy0 = xyz_rpy[:, 0]
        xyz_rpy_pred = self.lstm(xyz_rpy0, controls, hm)
        return xyz_rpy_pred

    def compute_loss(self, batch, xyz_rpy_pred):
        hm, control_ts, controls, traj_ts, xyz_rpy = batch
        # find the closest timesteps in the trajectory to the ground truth timesteps
        ts_ids = torch.argmin(torch.abs(control_ts.unsqueeze(1) - traj_ts.unsqueeze(2)), dim=2)
        # compute the loss as the mean squared error between the predicted and ground truth poses
        batch_size = xyz_rpy_pred.size(0)
        loss = self.criterion(xyz_rpy_pred[torch.arange(batch_size).unsqueeze(1), ts_ids], xyz_rpy)
        return loss

    def vis(self, batch, xyz_rpy_pred):
        with torch.no_grad():
            hm, control_ts, controls, traj_ts, xyz_rpy = batch

            fig = plt.figure(figsize=(21, 7))
            plt.subplot(1, 3, 1)
            plt.plot(xyz_rpy[0, :, 0].cpu().numpy(), xyz_rpy[0, :, 1].cpu().numpy(),
                     '-r', label='GT poses')
            plt.plot(xyz_rpy_pred[0, :, 0].cpu().numpy(), xyz_rpy_pred[0, :, 1].cpu().numpy(),
                     '--b', label='Pred poses')
            plt.grid()
            plt.axis('equal')
            plt.legend()
            plt.title('Trajectories XY')
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')

            plt.subplot(1, 3, 2)
            plt.plot(traj_ts[0].cpu().numpy(), xyz_rpy[0, :, 2].cpu().numpy(),
                     '-r', label='GT poses')
            plt.plot(control_ts[0].cpu().numpy(), xyz_rpy_pred[0, :, 2].cpu().numpy(),
                     '--b', label='Pred poses')
            plt.grid()
            plt.ylim(-1, 1)
            plt.legend()
            plt.title('Trajectories Z')
            plt.xlabel('t [s]')
            plt.ylabel('Z [m]')

            plt.subplot(1, 3, 3)
            plt.title('Euler angles')
            plt.plot(traj_ts[0].cpu().numpy(), xyz_rpy[0, :, 3].cpu().numpy(),
                     '-r', label='GT roll')
            plt.plot(control_ts[0].cpu().numpy(), xyz_rpy_pred[0, :, 3].cpu().numpy(),
                     '--r', label='Pred roll')
            plt.plot(traj_ts[0].cpu().numpy(), xyz_rpy[0, :, 4].cpu().numpy(),
                     '-g', label='GT pitch')
            plt.plot(control_ts[0].cpu().numpy(), xyz_rpy_pred[0, :, 4].cpu().numpy(),
                     '--g', label='Pred pitch')
            plt.plot(traj_ts[0].cpu().numpy(), xyz_rpy[0, :, 5].cpu().numpy(),
                     '-b', label='GT yaw')
            plt.plot(control_ts[0].cpu().numpy(), xyz_rpy_pred[0, :, 5].cpu().numpy(),
                     '--b', label='Pred yaw')
            plt.ylim(-1, 1)
            plt.legend()
            plt.grid()
            plt.xlabel('t [s]')
            plt.ylabel('angle [rad]')

        return fig

    def epoch(self, train=True):
        # choose data loader
        loader = self.train_loader if train else self.val_loader

        # set model mode
        if train:
            self.lstm.train()
        else:
            self.lstm.eval()

        # Training loop
        epoch_loss = 0.
        for batch in tqdm(loader, total=len(loader)):
            if train:
                self.optimizer.zero_grad()

            # model inference
            batch = [b.to(self.device) for b in batch]
            xyz_rpy_pred = self.inference(batch)

            # compute loss
            loss = self.compute_loss(batch, xyz_rpy_pred)
            epoch_loss += loss.item()

            if train:
                # backpropagation and optimization
                loss.backward()
                self.optimizer.step()

        epoch_loss /= len(loader)

        return epoch_loss

    def train(self):
        min_loss = np.inf
        for e in range(self.n_epochs):
            train_epoch_loss = self.epoch(train=True)
            print(f'Train epoch {e} loss: {train_epoch_loss}')
            self.writer.add_scalar('train/loss', train_epoch_loss, global_step=e)

            with torch.no_grad():
                val_epoch_loss = self.epoch(train=False)
                print(f'Val epoch {e} loss: {val_epoch_loss}')
                self.writer.add_scalar('val/loss', val_epoch_loss, global_step=e)

                # visualize predictions
                sample_i = np.random.randint(len(self.val_loader.dataset))
                sample = self.val_loader.dataset[sample_i]
                batch = [s[None].to(self.device) for s in sample]
                xyz_rpy_pred = self.inference(batch)
                fig = self.vis(batch, xyz_rpy_pred)
                self.writer.add_figure('val/sample', fig, global_step=e)

                # save best model
                if val_epoch_loss < min_loss:
                    print('Saving model...')
                    min_loss = val_epoch_loss
                    self.lstm.eval()
                    torch.save(self.lstm.state_dict(), os.path.join(self.log_dir, 'lstm.pth'))


def main():
    trainer = Trainer(batch_size=16, n_epochs=1000)
    trainer.train()


if __name__ == '__main__':
    main()
