#!/usr/bin/env python

import sys
sys.path.append('../src/')
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import DataLoader
from collections import deque
import argparse
from monoforce.models.physics_engine.engine.engine import DPhysicsEngine, PhysicsState
from monoforce.configs import WorldConfig, RobotModelConfig, PhysicsEngineConfig
from monoforce.models.physics_engine.engine.engine_state import vectorize_iter_of_states as vectorize_states
from monoforce.models.physics_engine.utils.environment import make_x_y_grids
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, denormalize_img
from monoforce.utils import read_yaml, write_to_csv, append_to_csv, compile_data, str2bool
from monoforce.losses import physics_loss, hm_loss


def arg_parser():
    parser = argparse.ArgumentParser(description='Terrain encoder predictor input arguments')
    parser.add_argument('--seq', type=str, default='val', help='Data sequence')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--pretrained_terrain_encoder_path', type=str, default=None,
                        help='Path to the trained LSS model')
    parser.add_argument('--vis', type=str2bool, default=True, help='Visualize the results')
    return parser.parse_args()


class Evaluator:
    def __init__(self,
                 batch_size: int = 1,
                 pretrained_terrain_encoder_path=None,
                 grid_res: float = 0.1,
                 max_coord: float = 6.4,
                 terrain_simplification_scale: int = 1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        # load configs
        self.robot_model = RobotModelConfig(kind='marv').to(self.device)
        x_grid, y_grid = make_x_y_grids(max_coord, grid_res, self.batch_size)
        z_grid = torch.zeros_like(x_grid)
        self.world_config = WorldConfig(
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=z_grid,
            grid_res=grid_res,
            max_coord=max_coord,
        ).to(self.device)
        self.physics_config = PhysicsEngineConfig(num_robots=batch_size).to(self.device)
        self.physics_engine = self.get_physics_engine()

        # load LSS config
        self.lss_config = read_yaml(os.path.join('..', 'config/lss_cfg.yaml'))
        self.terrain_encoder = self.get_terrain_encoder(pretrained_terrain_encoder_path)

        # terrain simplification for faster physics execution
        self.done_xy_grid_simplification = False  # simplification done only once for x and y grid
        self.grid_res = grid_res
        self.terrain_simplification_scale = terrain_simplification_scale
        self.terrain_preproc = torch.nn.AvgPool2d(kernel_size=terrain_simplification_scale,
                                                  stride=terrain_simplification_scale)

    def get_terrain_encoder(self, path):
        terrain_encoder = LiftSplatShoot(self.lss_config['grid_conf'],
                                         self.lss_config['data_aug_conf']).from_pretrained(path)
        terrain_encoder.to(self.device)
        return terrain_encoder

    def predict_terrain(self, batch):
        imgs, rots, trans, intrins, post_rots, post_trans = batch[:6]
        img_inputs = (imgs, rots, trans, intrins, post_rots, post_trans)
        terrain = self.terrain_encoder(*img_inputs)
        return terrain

    def get_physics_engine(self):
        enine = DPhysicsEngine(self.physics_config, self.robot_model, self.device)
        enine.to(self.device)
        return enine

    def predict_states(self, terrain, batch):
        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain,
         control_ts, controls,
         traj_ts, xs, xds, qs, omegas, thetas) = batch
        height, friction = terrain['terrain'], terrain['friction']
        n_trajs, n_iters = controls.shape[:2]

        # Initial state
        x0 = xs[:, 0].contiguous()
        xd0 = xds[:, 0].contiguous()
        q0 = qs[:, 0].contiguous()
        omega0 = omegas[:, 0].contiguous()
        thetas0 = thetas[:, 0].contiguous()
        state0 = PhysicsState(x0, xd0, q0, omega0, thetas0, batch_size=x0.shape[0])

        if not self.done_xy_grid_simplification:
            # terrain simplification for faster physics execution
            self.world_config.x_grid = self.terrain_preproc(self.world_config.x_grid)
            self.world_config.y_grid = self.terrain_preproc(self.world_config.y_grid)
            self.world_config.grid_res = self.terrain_simplification_scale * self.grid_res
            self.done_xy_grid_simplification = True
        self.world_config.z_grid = self.terrain_preproc(height.squeeze(1))
        states_pred = deque(maxlen=n_iters)
        state = state0
        for i in range(n_iters):
            state, der, aux = self.physics_engine(state, controls[:, i], self.world_config)
            states_pred.append(state)
        states_pred = vectorize_states(states_pred)

        return states_pred

    @torch.inference_mode()
    def run(self, vis=False):
        # set models to eval mode
        self.terrain_encoder.eval()
        self.physics_engine.eval()

        # output folder to write evaluation results
        self.output_folder = (f'./gen/eval/'
                              f'{self.terrain_encoder.__class__.__name__}_'
                              f'{self.physics_engine.__class__.__name__}')

        # load dataset
        _, val_ds = compile_data(lss_cfg=self.lss_config)
        loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # create output folder
        os.makedirs(self.output_folder, exist_ok=True)
        # write losses to output csv
        write_to_csv(f'{self.output_folder}/losses.csv', 'Batch i,H_g loss,H_t loss,XYZ loss\n')

        H, W = self.lss_config['data_aug_conf']['H'], self.lss_config['data_aug_conf']['W']
        cams = ['cam_left', 'cam_front', 'cam_right', 'cam_rear']

        x_grid = self.world_config.x_grid[0].cpu()
        y_grid = self.world_config.y_grid[0].cpu()

        fig, axes = plt.subplots(3, 4, figsize=(20, 16))
        for i, batch in enumerate(tqdm(loader)):
            batch = [t.to(self.device) for t in batch]
            # get a sample from the dataset
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_geom, hm_terrain,
             control_ts, controls,
             traj_ts, xs, xds, qs, omegas, thetas) = batch
            states_gt = [xs, xds, qs, omegas, thetas]

            # terrain prediction
            terrain = self.predict_terrain(batch)
            H_t_pred, H_g_pred, H_diff_pred, Friction_pred = terrain['terrain'], terrain['geom'], terrain['diff'], \
            terrain['friction']

            # terrain and geom heightmap losses
            loss_geom = hm_loss(height_pred=H_g_pred[:, 0], height_gt=hm_geom[:, 0], weights=hm_geom[:, 1])
            loss_terrain = hm_loss(height_pred=H_t_pred[:, 0], height_gt=hm_terrain[:, 0], weights=hm_terrain[:, 1])

            # trajectory prediction loss: xyz and rotation
            states_pred = self.predict_states(terrain, batch)
            loss_xyz = physics_loss(states_pred=[states_pred.x.permute(1, 0, 2)], states_gt=states_gt,
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
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_geom, hm_terrain,
             control_ts, controls,
             traj_ts, xs, xds, qs, omegas, thetas) = batch
            states_gt = [xs, xds, qs, omegas, thetas]

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
            axes[2, 0].plot(control_ts[0], controls[0, :, 0], 'g', label='v_(t)')
            axes[2, 0].plot(control_ts[0], controls[0, :, 1], 'b', label='v_(t)')
            axes[2, 0].plot(control_ts[0], controls[0, :, 2], 'g--', label='v_(t)')
            axes[2, 0].plot(control_ts[0], controls[0, :, 3], 'b--', label='v_(t)')
            axes[2, 0].grid()
            axes[2, 0].set_xlabel('Time [s]')
            axes[2, 0].set_ylabel('Flipper Vels [m/s]')
            axes[2, 0].legend()

            # plot trajectories: Roll, Pitch, Yaw
            rpy = Rotation.from_quat(states_pred.q[:, 0].cpu()).as_euler('xyz')
            rpy_gt = Rotation.from_quat(states_gt[2][0].cpu()).as_euler('xyz')
            axes[2, 1].plot(control_ts[0], rpy[:, 0], 'r', label='Pred Roll')
            axes[2, 1].plot(control_ts[0], rpy[:, 1], 'g', label='Pred Pitch')
            axes[2, 1].plot(control_ts[0], rpy[:, 2], 'b', label='Pred Yaw')
            axes[2, 1].plot(traj_ts[0], rpy_gt[:, 0], 'r--', label='Roll')
            axes[2, 1].plot(traj_ts[0], rpy_gt[:, 1], 'g--', label='Pitch')
            axes[2, 1].plot(traj_ts[0], rpy_gt[:, 2], 'b--', label='Yaw')
            axes[2, 1].grid()
            axes[2, 1].set_xlabel('Time [s]')
            axes[2, 1].set_ylabel('Angle [rad]')
            axes[2, 1].set_ylim(-np.pi / 2., np.pi / 2.)
            # axes[2, 1].legend()

            # plot trajectories: XY
            axes[2, 2].plot(states_pred.x[:, 0, 0].cpu(), states_pred.x[:, 0, 1].cpu(), 'r', label='Pred Traj')
            axes[2, 2].plot(states_gt[0][0, :, 0], states_gt[0][0, :, 1], 'k', label='GT Traj')
            axes[2, 2].set_xlim(-self.world_config.max_coord, self.world_config.max_coord)
            axes[2, 2].set_ylim(-self.world_config.max_coord, self.world_config.max_coord)
            # axes[2, 2].set_aspect('equal')
            axes[2, 2].grid()
            axes[2, 2].set_xlabel('x [m]')
            axes[2, 2].set_ylabel('y [m]')
            axes[2, 2].legend()

            # plot trajectories: Z
            axes[2, 3].plot(control_ts[0], states_pred.x[:, 0, 2].cpu(), 'r', label='Pred Traj')
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
    evaluator = Evaluator(batch_size=args.batch_size,
                          pretrained_terrain_encoder_path=args.pretrained_terrain_encoder_path)
    evaluator.run(vis=args.vis)


if __name__ == '__main__':
    main()