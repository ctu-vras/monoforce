import torch
from monoforce.models.terrain_encoder.bevfusion import BEVFusion
from monoforce.utils import read_yaml
from monoforce.datasets.rough import ROUGHPoints, rough_seq_paths


def test_bevfusion():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    lss_config = read_yaml('../config/lss_cfg.yaml')
    ds = ROUGHPoints(path=rough_seq_paths['marv'][0], lss_cfg=lss_config)
    print(f'Dataset length: {len(ds)}')

    model = BEVFusion(grid_conf=lss_config['grid_conf'], data_aug_conf=lss_config['data_aug_conf'])
    print(f'Number of BEVFusion model parameters: {sum(p.numel() for p in model.parameters())}')

    sample = ds[0]
    (imgs, rots, trans, intrins, post_rots, post_trans,
     hm_geom, hm_terrain,
     control_ts, controls,
     traj_ts, Xs, Xds, Rs, Omegas,
     points) = sample

    img_inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
    img_inputs = [torch.tensor(i, device=device).unsqueeze(0) for i in img_inputs]

    x = model.get_voxels(*img_inputs)
    print(f'IMG Voxel shape: {x.shape}')
    x = model.bevencode.backbone(x)
    print(f'Backbone output shape: {x.shape}')
    x = model.bevencode.up_geom(x)
    print(f'Geom head output shape: {x.shape}')

    cloud_input = torch.tensor(points, device=device).unsqueeze(0)
    print(f'LiDAR input shape: {cloud_input.shape}')
    x = model.lidar_bev.voxelize(cloud_input)
    print(f'Voxelized grid shape: {x.shape}')
    x = model.lidar_bev.lidar_encoder(x)
    print(f'LiDAR encoder output shape: {x.shape}')
    x = model.lidar_bev.bev_flatten(x)
    print(f'BEV features shape: {x.shape}')



def main():
    test_bevfusion()


if __name__ == '__main__':
    main()
