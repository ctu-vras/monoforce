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
    print(f'Voxel shape: {x.shape}')
    x = model.bevencode.backbone(x)
    print(f'Backbone output shape: {x.shape}')
    x = model.bevencode.up_geom(x)
    print(f'Geom head output shape: {x.shape}')

    cloud_input = torch.tensor(points, device=device).unsqueeze(0)
    print(f'LiDAR input shape: {cloud_input.shape}')

    x = model.lidar_bev(cloud_input)
    print(f'LiDAR BEV output shape: {x.shape}')


def test_voxelization():
    from monoforce.models.terrain_encoder.bevfusion import LiDARToBEV

    # Create a batch of random point clouds (B, 3, N), where B = 2 and N = 1000
    point_clouds = torch.rand(2, 3, 1000) * 100  # Two point clouds, 1000 points each

    lss_config = read_yaml('../config/lss_cfg.yaml')
    x_bound, y_bound, z_bound = lss_config['grid_conf']['xbound'], lss_config['grid_conf']['ybound'], lss_config['grid_conf']['zbound']

    # Define voxel size and grid size
    voxel_size = x_bound[2]
    D = int((z_bound[1] - z_bound[0]) / voxel_size)
    H, W = int((x_bound[1] - x_bound[0]) / voxel_size), int((y_bound[1] - y_bound[0]) / voxel_size)
    grid_size = (D, H, W)  # (D, H, W)

    # Initialize the model
    model = LiDARToBEV(voxel_size=voxel_size, grid_size=grid_size)

    # Forward pass: generate BEV features from the point cloud
    bev_output = model(point_clouds)
    print(f"BEV output shape: {bev_output.shape}")  # Expected output: (B, out_channels, H, W)


def main():
    test_bevfusion()
    # test_voxelization()


if __name__ == '__main__':
    main()
