import torch
import torch.nn as nn
from .lss import LiftSplatShoot


class LiDAREncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(LiDAREncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class BEVFlatten(nn.Module):
    def __init__(self):
        super(BEVFlatten, self).__init__()

    def forward(self, lidar_features):
        # Max-pooling along the z-axis to create 2D BEV features
        bev_features = torch.max(lidar_features, dim=2)[0]  # Max-pooling along z-axis
        return bev_features


class LiDARToBEV(nn.Module):
    def __init__(self, voxel_size, grid_size, out_channels=64):
        super(LiDARToBEV, self).__init__()
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.lidar_encoder = LiDAREncoder(in_channels=1, out_channels=out_channels)
        self.bev_flatten = BEVFlatten()

    def voxelize(self, point_clouds):
        """
        Voxelizes batched point clouds into a 3D grid.

        Args:
            point_clouds (torch.Tensor): The input point clouds of shape (B, 3, N),
                                         where B is the batch size, N is the number of points.
            voxel_size (float): The size of each voxel.
            grid_size (tuple): The size of the voxel grid (D, H, W).

        Returns:
            voxel_grid (torch.Tensor): A voxelized grid of shape (B, D, H, W).
        """
        # Step 1: Find the minimum point per batch in the point cloud
        min_bound = point_clouds.min(dim=2, keepdim=True)[0]  # Shape (B, 3, 1)

        # Step 2: Subtract the minimum point and scale by voxel size
        shifted_points = (point_clouds - min_bound) / self.voxel_size  # Normalize and scale

        # Step 3: Floor the points to get voxel indices
        grid_indices = torch.floor(shifted_points).long()  # Shape (B, 3, N)

        # Step 4: Clip indices to be within the voxel grid size
        grid_indices = torch.clamp(grid_indices, 0, self.grid_size[0] - 1)  # Clip along D
        grid_indices = torch.clamp(grid_indices, 0, self.grid_size[1] - 1)  # Clip along H
        grid_indices = torch.clamp(grid_indices, 0, self.grid_size[2] - 1)  # Clip along W

        # Step 5: Create a batch of voxel grids and mark occupied voxels
        B = point_clouds.shape[0]
        voxel_grid = torch.zeros((B, *self.grid_size), dtype=torch.float32)  # (B, D, H, W)

        for b in range(B):
            batch_indices = grid_indices[b]  # Get indices for the b-th batch
            voxel_grid[b, batch_indices[0, :], batch_indices[1, :], batch_indices[2, :]] = 1.0

        voxel_grid = voxel_grid.unsqueeze(1)  # Add channel dimension

        return voxel_grid

    def forward(self, point_cloud):
        # Step 1: Voxelize the raw point cloud
        voxel_grid = self.voxelize(point_cloud)

        # Step 2: Encode the voxelized point cloud using 3D CNN
        lidar_features = self.lidar_encoder(voxel_grid)

        # Step 3: Flatten along the z-axis to obtain BEV features
        bev_features = self.bev_flatten(lidar_features)

        return bev_features

class BEVFusion(LiftSplatShoot):
    def __init__(self, grid_conf, data_aug_conf):
        super().__init__(grid_conf, data_aug_conf)

        # Initialize the model
        voxel_size, grid_size = self.get_vox_dims()
        self.lidar_bev = LiDARToBEV(voxel_size=voxel_size, grid_size=grid_size)

    def get_vox_dims(self):
        # Define voxel size and grid size
        x_bound = self.grid_conf['xbound']
        y_bound = self.grid_conf['ybound']
        z_bound = self.grid_conf['zbound']
        voxel_size = x_bound[2]
        D = int((z_bound[1] - z_bound[0]) / voxel_size)
        H, W = int((x_bound[1] - x_bound[0]) / voxel_size), int((y_bound[1] - y_bound[0]) / voxel_size)
        grid_size = (D, H, W)  # (D, H, W)
        return voxel_size, grid_size
