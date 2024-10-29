#!/usr/bin/env python

import sys
sys.path.append('../src/')
import torch
import numpy as np
from torch.utils.data import DataLoader
from monoforce.models.terrain_encoder.bevfusion import BEVFusion
from monoforce.utils import read_yaml, position
from monoforce.datasets.rough import ROUGH, rough_seq_paths
from monoforce.dphys_config import DPhysConfig
from monoforce.transformations import transform_cloud


class Data(ROUGH):
    def __init__(self, path, lss_cfg, dphys_cfg=DPhysConfig(), is_train=True):
        super(Data, self).__init__(path, lss_cfg, dphys_cfg=dphys_cfg, is_train=is_train)

    def get_cloud(self, i, points_source='lidar'):
        cloud = self.get_raw_cloud(i)
        # move points to robot frame
        Tr = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr = np.asarray(Tr, dtype=float).reshape((4, 4))
        cloud = transform_cloud(cloud, Tr)
        return cloud

    def get_sample(self, i):
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_images_data(i)
        control_ts, controls = self.get_controls(i)
        traj_ts, states = self.get_states_traj(i)
        Xs, Xds, Rs, Omegas = states
        hm_geom = self.get_geom_height_map(i)
        hm_terrain = self.get_terrain_height_map(i)
        points = torch.as_tensor(position(self.get_cloud(i))).T
        return (imgs, rots, trans, intrins, post_rots, post_trans,
                hm_geom, hm_terrain,
                control_ts, controls,
                traj_ts, Xs, Xds, Rs, Omegas,
                points)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lss_config = read_yaml('../config/lss_cfg.yaml')
    ds = Data(path=rough_seq_paths['marv'][0], lss_cfg=lss_config)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    print(f'Dataset length: {len(ds)}')

    bevfusion = BEVFusion(grid_conf=lss_config['grid_conf'], data_aug_conf=lss_config['data_aug_conf'])
    bevfusion.to(device)
    print(f'Number of BEVFusion model parameters: {sum(p.numel() for p in bevfusion.parameters())}')

    batch = next(iter(loader))
    (imgs, rots, trans, intrins, post_rots, post_trans,
     hm_geom, hm_terrain,
     control_ts, controls,
     traj_ts, Xs, Xds, Rs, Omegas,
     points) = batch

    img_inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
    img_inputs = [i.to(device) for i in img_inputs]
    points_input = points.to(device)

    with torch.inference_mode():
        with torch.no_grad():
            out = bevfusion(img_inputs, points_input)
            for k, v in out.items():
                print(f'{k}: {v.shape}')


if __name__ == '__main__':
    main()
