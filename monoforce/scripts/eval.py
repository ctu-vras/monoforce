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
from collections import deque
import argparse
from monoforce.models.physics_engine.engine.engine import DPhysicsEngine, PhysicsState
from monoforce.models.physics_engine.configs import WorldConfig, RobotModelConfig, PhysicsEngineConfig
from monoforce.models.physics_engine.engine.engine_state import vectorize_iter_of_states as vectorize_states
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, denormalize_img
from monoforce.utils import read_yaml, write_to_csv, append_to_csv, compile_data, str2bool
from monoforce.losses import physics_loss, hm_loss
from monoforce.datasets import ROUGH, rough_seq_paths


def arg_parser():
    parser = argparse.ArgumentParser(description='Terrain encoder predictor input arguments')
    parser.add_argument('--seq', type=str, default='val', help='Data sequence')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--terrain_encoder', type=str, default='lss', help='Terrain encoder model')
    parser.add_argument('--terrain_encoder_path', type=str, default=None, help='Path to the LSS model')
    parser.add_argument('--vis', type=str2bool, default=True, help='Visualize the results')
    return parser.parse_args()


class Eval:
    def __init__(self,
                 seq='val',
                 batch_size=1,
                 terrain_encoder='lss',
                 terrain_encoder_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load DPhys config
        if seq in rough_seq_paths:
            robot = os.path.basename(seq).split('_')[0]
            robot = 'tradr' if robot == 'ugv' else 'marv'
        else:
            robot = 'marv'
        print(f'Robot: {robot}')
        self.robot_model = RobotModelConfig(kind=robot).to(self.device)
        grid_res = 0.1  # 10cm per grid cell
        max_coord = 6.4  # meters
        DIM = int(2 * max_coord / grid_res)
        xint = torch.linspace(-max_coord, max_coord, DIM)
        yint = torch.linspace(-max_coord, max_coord, DIM)
        x_grid, y_grid = torch.meshgrid(xint, yint, indexing="xy")  # use torch's XY indexing !!!!!
        z_grid = torch.zeros_like(x_grid)
        self.world_config = WorldConfig(
            x_grid=x_grid.repeat(batch_size, 1, 1),
            y_grid=y_grid.repeat(batch_size, 1, 1),
            z_grid=z_grid.repeat(batch_size, 1, 1),
            grid_res=grid_res,
            max_coord=max_coord,
        ).to(self.device)
        self.physics_config = PhysicsEngineConfig(num_robots=batch_size).to(self.device)
        self.physics_engine = self.get_physics_engine()

        # load LSS config
        self.lss_config = read_yaml(os.path.join('..', 'config/lss_cfg.yaml'))
        self.terrain_encoder = self.get_terrain_encoder(terrain_encoder_path, model=terrain_encoder)

        # load data
        self.loader = self.get_dataloader(batch_size=batch_size, seq=seq)

        # output folder to write evaluation results
        self.output_folder = (f'./gen/eval_{os.path.basename(seq)}/'
                              f'{robot}_{self.terrain_encoder.__class__.__name__}_'
                              f'{self.physics_engine.__class__.__name__}')

    def get_terrain_encoder(self, path, model='lss'):
        if model == 'lss':
            terrain_encoder = LiftSplatShoot(self.lss_config['grid_conf'],
                                             self.lss_config['data_aug_conf']).from_pretrained(path)
        else:
            raise ValueError(f'Invalid terrain encoder model: {model}. Supported: lss')
        terrain_encoder.to(self.device)
        terrain_encoder.eval()
        return terrain_encoder

    def predict_terrain(self, batch):
        model = self.terrain_encoder.__class__.__name__
        if model == 'LiftSplatShoot':
            imgs, rots, trans, intrins, post_rots, post_trans = batch[:6]
            img_inputs = (imgs, rots, trans, intrins, post_rots, post_trans)
            terrain = self.terrain_encoder(*img_inputs)
        else:
            raise ValueError(f'Invalid terrain encoder model: {model}. Supported: LiftSplatShoot')
        return terrain

    def get_physics_engine(self):
        enine = DPhysicsEngine(self.physics_config, self.robot_model, self.device)
        enine.to(self.device)
        enine.eval()
        return enine

    def predict_states(self, terrain, batch):
        Xs, Xds, Rs, Omegas = batch[12:16]
        vws = batch[9]
        # state0 = tuple([s[:, 0] for s in [Xs, Xds, Rs, Omegas]])
        height, friction = terrain['terrain'], terrain['friction']

        # convert vws to controls and add flipper controls
        n_trajs, n_iters = vws.shape[:2]
        track_vels = self.robot_model.vw_to_vels(v=vws[..., 0].view(-1,), w=vws[..., 1].view(-1,))
        track_vels = track_vels.reshape(n_trajs, n_iters, -1)
        flipper_controls = torch.zeros_like(track_vels)
        controls = torch.cat((track_vels, flipper_controls), dim=-1)

        # Initial state
        x0 = Xs[:, 0]
        xd0 = Xds[:, 0]
        q0 = torch.as_tensor(Rotation.from_matrix(Rs[:, 0].cpu()).as_quat(), dtype=torch.float32).to(self.device)
        omega0 = Omegas[:, 0]
        thetas0 = torch.zeros(n_trajs, self.robot_model.num_driving_parts).to(self.device)
        state0 = PhysicsState(x0, xd0, q0, omega0, thetas0)

        self.world_config.z_grid = height.squeeze(1)
        states_pred = deque(maxlen=n_iters)
        state = state0
        for i in range(n_iters):
            state, der, aux = self.physics_engine(state, controls[:, i], self.world_config)
            states_pred.append(state)
        states_pred = vectorize_states(states_pred)

        return states_pred

    def get_dataloader(self, batch_size=1, seq='val'):
        if seq != 'val':
            print('Loading dataset from:', seq)
            val_ds = ROUGH(path=seq, lss_cfg=self.lss_config)
        else:
            _, val_ds = compile_data(lss_cfg=self.lss_config)
        loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return loader

    @torch.inference_mode()
    def run(self, vis=False):
        # create output folder
        os.makedirs(self.output_folder, exist_ok=True)
        # write losses to output csv
        write_to_csv(f'{self.output_folder}/losses.csv', 'Batch i,H_g loss,H_t loss,XYZ loss\n')

        H, W = self.lss_config['data_aug_conf']['H'], self.lss_config['data_aug_conf']['W']
        cams = ['cam_left', 'cam_front', 'cam_right', 'cam_rear']

        x_grid = self.world_config.x_grid[0].cpu()
        y_grid = self.world_config.y_grid[0].cpu()

        fig, axes = plt.subplots(3, 4, figsize=(20, 16))
        for i, batch in enumerate(tqdm(self.loader)):
            batch = [t.to(self.device) for t in batch]
            # get a sample from the dataset
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_geom, hm_terrain,
             control_ts, controls,
             pose0,
             traj_ts, Xs, Xds, Rs, Omegas) = batch
            states_gt = [Xs, Xds, Rs, Omegas]

            # terrain prediction
            terrain = self.predict_terrain(batch)
            H_t_pred, H_g_pred, H_diff_pred, Friction_pred = terrain['terrain'], terrain['geom'], terrain['diff'], \
            terrain['friction']

            # terrain and geom heightmap losses
            loss_geom = hm_loss(height_pred=H_g_pred[:, 0], height_gt=hm_geom[:, 0], weights=hm_geom[:, 1])
            loss_terrain = hm_loss(height_pred=H_t_pred[:, 0], height_gt=hm_terrain[:, 0], weights=hm_terrain[:, 1])

            # trajectory prediction loss: xyz and rotation
            states_pred = self.predict_states(terrain, batch)
            states_pred = [states_pred.x.permute(1, 0, 2)]
            loss_xyz = physics_loss(states_pred=states_pred, states_gt=states_gt,
                                    pred_ts=control_ts, gt_ts=traj_ts,
                                    gamma=1.0)

            # write losses to csv
            append_to_csv(f'{self.output_folder}/losses.csv',
                          f'{i:04d}, {loss_geom.item()},{loss_terrain.item()},{loss_xyz.item()}\n')

            # visualizations
            H_g_pred = H_g_pred[0, 0].cpu()
            H_diff_pred = H_diff_pred[0, 0].cpu()
            H_t_pred = H_t_pred[0, 0].cpu()
            Friction_pred = Friction_pred[0, 0].cpu()
            # get height map points
            hm_points = torch.stack([x_grid, y_grid, H_t_pred], dim=-1)
            hm_points = hm_points.view(-1, 3).T

            batch = [t.to('cpu') for t in batch]
            # get a sample from the dataset
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_geom, hm_terrain,
             control_ts, controls,
             pose0,
             traj_ts, Xs, Xds, Rs, Omegas) = batch
            states_gt = [Xs, Xds, Rs, Omegas]

            # clear axis
            for ax in axes.flatten():
                ax.clear()
            plt.suptitle(f'Terrain Loss: {loss_terrain.item():.3f}, Traj Loss: {loss_xyz.item():.3f}')
            for imgi, img in enumerate(imgs[0]):
                ax = axes[0, imgi]

                cam_pts = ego_to_cam(hm_points, rots[0, imgi], trans[0, imgi], intrins[0, imgi])
                mask = get_only_in_img_mask(cam_pts, H, W)
                plot_pts = post_rots[0, imgi].matmul(cam_pts) + post_trans[0, imgi].unsqueeze(1)
                showimg = denormalize_img(img)

                ax.imshow(showimg)
                ax.scatter(plot_pts[0, mask], plot_pts[1, mask],
                           # c=Friction_pred.view(-1)[terrain_mask][mask],
                           c=hm_points[2, mask],
                           s=2, alpha=0.8, cmap='jet', vmin=-1, vmax=1.)
                ax.axis('off')
                # camera name as text on image
                ax.text(0.5, 0.9, cams[imgi].replace('_', ' '),
                        horizontalalignment='center', verticalalignment='top',
                        transform=ax.transAxes, fontsize=10)

            # plot geom heightmap
            axes[1, 0].set_title('Geom Height')
            axes[1, 0].imshow(H_g_pred, origin='lower', cmap='jet', vmin=-1., vmax=1.)
            axes[1, 0].axis('off')

            # plot height diff heightmap
            axes[1, 1].set_title('Height Difference')
            axes[1, 1].imshow(H_diff_pred, origin='lower', cmap='jet', vmin=-1., vmax=1.)
            axes[1, 1].axis('off')

            # plot terrain heightmap
            axes[1, 2].set_title('Terrain Height')
            axes[1, 2].imshow(H_t_pred, origin='lower', cmap='jet', vmin=-1., vmax=1.)
            axes[1, 2].axis('off')

            # plot friction map
            axes[1, 3].set_title('Friction')
            axes[1, 3].imshow(Friction_pred, origin='lower', cmap='jet', vmin=0., vmax=1.)
            axes[1, 3].axis('off')

            # plot control inputs
            axes[2, 0].plot(control_ts[0], controls[0, :, 0], c='g', label='v(t)')
            axes[2, 0].plot(control_ts[0], controls[0, :, 1], c='b', label='w(t)')
            axes[2, 0].grid()
            axes[2, 0].set_xlabel('Time [s]')
            axes[2, 0].set_ylabel('Control [m/s]')
            axes[2, 0].legend()

            # # plot trajectories: Roll, Pitch, Yaw
            # rpy = Rotation.from_matrix(states_pred[2][0].cpu()).as_euler('xyz')
            # rpy_gt = Rotation.from_matrix(states_gt[2][0].cpu()).as_euler('xyz')
            # axes[2, 1].plot(control_ts[0], rpy[:, 0], 'r', label='Pred Roll')
            # axes[2, 1].plot(control_ts[0], rpy[:, 1], 'g', label='Pred Pitch')
            # axes[2, 1].plot(control_ts[0], rpy[:, 2], 'b', label='Pred Yaw')
            # axes[2, 1].plot(traj_ts[0], rpy_gt[:, 0], 'r--', label='Roll')
            # axes[2, 1].plot(traj_ts[0], rpy_gt[:, 1], 'g--', label='Pitch')
            # axes[2, 1].plot(traj_ts[0], rpy_gt[:, 2], 'b--', label='Yaw')
            # axes[2, 1].grid()
            # axes[2, 1].set_xlabel('Time [s]')
            # axes[2, 1].set_ylabel('Angle [rad]')
            # axes[2, 1].set_ylim(-np.pi / 2., np.pi / 2.)
            # # axes[2, 1].legend()

            # plot trajectories: XY
            axes[2, 2].plot(states_pred[0][0, :, 0].cpu(), states_pred[0][0, :, 1].cpu(), 'r', label='Pred Traj')
            axes[2, 2].plot(states_gt[0][0, :, 0], states_gt[0][0, :, 1], 'k', label='GT Traj')
            axes[2, 2].set_xlim(-self.world_config.max_coord, self.world_config.max_coord)
            axes[2, 2].set_ylim(-self.world_config.max_coord, self.world_config.max_coord)
            axes[2, 2].grid()
            axes[2, 2].set_xlabel('x [m]')
            axes[2, 2].set_ylabel('y [m]')
            axes[2, 2].legend()

            # plot trajectories: Z
            axes[2, 3].plot(control_ts[0], states_pred[0][0, :, 2].cpu(), 'r', label='Pred Traj')
            axes[2, 3].plot(traj_ts[0], states_gt[0][0, :, 2], 'k', label='GT Traj')
            axes[2, 3].grid()
            axes[2, 3].set_xlabel('Time [s]')
            axes[2, 3].set_ylabel('z [m]')
            axes[2, 3].set_ylim(-1.0, 1.0)
            axes[2, 3].legend()

            if vis:
                plt.pause(0.01)
                plt.draw()
            plt.savefig(f'{self.output_folder}/{i:04d}.png')
        plt.close(fig)


def main():
    args = arg_parser()
    print(args)
    eval = Eval(seq=args.seq,
                batch_size=args.batch_size,
                terrain_encoder=args.terrain_encoder,
                terrain_encoder_path=args.terrain_encoder_path)
    eval.run(vis=args.vis)


if __name__ == '__main__':
    main()