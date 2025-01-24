import torch
from torch import nn as nn
from .lss import BevEncode


class LidarNet(nn.Module):
    def __init__(self, grid_conf, in_channels=1, out_channels=64):
        super(LidarNet, self).__init__()
        assert grid_conf['xbound'][2] == grid_conf['ybound'][2], 'Voxel size must be the same in x and y dimensions'
        self.grid_conf = grid_conf
        self.voxel_size = grid_conf['xbound'][2]
        self.grid_size = (int((grid_conf['xbound'][1] - grid_conf['xbound'][0]) / self.voxel_size),
                          int((grid_conf['ybound'][1] - grid_conf['ybound'][0]) / self.voxel_size),
                          int((grid_conf['zbound'][1] - grid_conf['zbound'][0]) / self.voxel_size))  # (X, Y, Z)
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )

    def voxelize(self, point_clouds):
        """
        Voxelizes batched point clouds into a 3D grid.

        Args:
            point_clouds (torch.Tensor): The input point clouds of shape (B, 3, N),
                                         where B is the batch size, N is the number of points.
        Returns:
            voxel_grid (torch.Tensor): A voxelized grid of shape (B, 1, Z, Y, X),
                                        where Z, Y, X are the dimensions of the grid.

        Example:
            Grid size: (X, Y, Z) = (64, 128, 128) means 64 height voxels, 128 width voxels, 128 depth voxels
            Voxel is a cube with size 0.1m x 0.1m x 0.1m

            >>> point_clouds = torch.rand(2, 3, 1000)  # 2 batches, 3D points, 1000 points
            >>> voxel_grid = self.voxelize(point_clouds)
            >>> voxel_grid.shape  # (2, 1, 64, 128, 128)
        """
        # Remove nans saving dimensions
        point_clouds[torch.isnan(point_clouds)] = 0

        # Find the minimum point per batch in the point cloud
        min_bound = torch.tensor([self.grid_conf['xbound'][0],
                                  self.grid_conf['ybound'][0],
                                  self.grid_conf['zbound'][0]], device=point_clouds.device).unsqueeze(0).unsqueeze(2)  # Shape (1, 3, 1)
        assert min_bound.shape == (1, 3, 1), f"min_bound shape is {min_bound.shape} != (1, 3, 1)"

        # Subtract the minimum point and scale by voxel size
        shifted_points = (point_clouds - min_bound) / self.voxel_size  # Normalize and scale

        # Floor the points to get voxel indices
        grid_indices = torch.floor(shifted_points).long()  # Shape (B, 3, N)

        # Create a batch of voxel grids and mark occupied voxels
        B = point_clouds.shape[0]
        voxel_grid = torch.zeros((B, *self.grid_size), device=point_clouds.device, dtype=torch.float32)  # (B, X, Y, Z)
        # TODO: get rid of the for loop, use torch.scatter possibly
        for b in range(B):
            mask_x = (grid_indices[b, 0] >= 0) & (grid_indices[b, 0] < self.grid_size[0])
            mask_y = (grid_indices[b, 1] >= 0) & (grid_indices[b, 1] < self.grid_size[1])
            mask_z = (grid_indices[b, 2] >= 0) & (grid_indices[b, 2] < self.grid_size[2])
            mask = mask_x & mask_y & mask_z
            grid_indices_b = grid_indices[b, :, mask]
            voxel_grid[b, grid_indices_b[0], grid_indices_b[1], grid_indices_b[2]] = 1

        # Permute the dimensions to (B, Z, X, Y)
        voxel_grid = voxel_grid.permute(0, 3, 1, 2)

        # Add channel dimension, (B, 1, Z, X, Y)
        voxel_grid = voxel_grid.unsqueeze(1)

        return voxel_grid

    def bev_flatten(self, lidar_features):
        # Pooling along the z-axis to create 2D BEV features
        bev_features = lidar_features.mean(dim=2)
        return bev_features

    def forward(self, point_cloud):
        # Step 1: Voxelize the raw point cloud
        voxel_grid = self.voxelize(point_cloud)  # Shape (B, 1, Z, X, Y)

        # Step 2: Encode the voxelized point cloud using 3D CNN
        lidar_features = self.encoder(voxel_grid)  # Shape (B, out_channels, Z, X, Y)

        # Step 3: Flatten along the z-axis to obtain BEV features
        bev_features = self.bev_flatten(lidar_features)  # Shape (B, out_channels, X, Y)

        return bev_features


class VoxelNet(nn.Module):
    def __init__(self, grid_conf, n_features=16, outC=1):
        super().__init__()

        self.lidar_net = LidarNet(grid_conf=grid_conf, out_channels=n_features)
        self.bevencode = BevEncode(inC=n_features, outC=outC)

    def forward(self, points):
        # Get features from LiDAR inputs
        x = self.lidar_net(points)  # Shape (B, C, X, Y)

        # Encode the BEV features
        out = self.bevencode(x)  # Shape (B, 1, X, Y)

        return out

    def from_pretrained(self, modelf):
        if not modelf:
            return self
        print(f'Loading pretrained {self.__class__.__name__} model from', modelf)
        # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
        model_dict = self.state_dict()
        pretrained_model = torch.load(modelf)
        model_dict.update(pretrained_model)
        self.load_state_dict(model_dict)
        return self
