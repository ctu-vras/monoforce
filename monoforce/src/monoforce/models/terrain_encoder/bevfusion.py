import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18
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


class LiDARBEV(nn.Module):
    def __init__(self, grid_conf, out_channels=64):
        super(LiDARBEV, self).__init__()
        assert grid_conf['xbound'][2] == grid_conf['ybound'][2], 'Voxel size must be the same in x and y dimensions'
        self.grid_conf = grid_conf
        self.voxel_size = grid_conf['xbound'][2]
        self.grid_size = (int((grid_conf['xbound'][1] - grid_conf['xbound'][0]) / self.voxel_size),
                          int((grid_conf['ybound'][1] - grid_conf['ybound'][0]) / self.voxel_size),
                          int((grid_conf['zbound'][1] - grid_conf['zbound'][0]) / self.voxel_size))  # (D, H, W)
        self.encoder = LiDAREncoder(in_channels=1, out_channels=out_channels)

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
        voxel_grid = torch.zeros((B, *self.grid_size), device=point_clouds.device, dtype=torch.float32)  # (B, H, W, D)
        # TODO: get rid of the for loop, use torch.scatter possibly
        for b in range(B):
            mask_x = (grid_indices[b, 0] >= 0) & (grid_indices[b, 0] < self.grid_size[0])
            mask_y = (grid_indices[b, 1] >= 0) & (grid_indices[b, 1] < self.grid_size[1])
            mask_z = (grid_indices[b, 2] >= 0) & (grid_indices[b, 2] < self.grid_size[2])
            mask = mask_x & mask_y & mask_z
            grid_indices_b = grid_indices[b, :, mask]
            voxel_grid[b, grid_indices_b[0], grid_indices_b[1], grid_indices_b[2]] = 1

        # Permute the dimensions to (B, D, H, W)
        voxel_grid = voxel_grid.permute(0, 3, 1, 2)

        # Add channel dimension, (B, 1, D, H, W)
        voxel_grid = voxel_grid.unsqueeze(1)

        return voxel_grid

    def bev_flatten(self, lidar_features):
        # Max-pooling along the z-axis to create 2D BEV features
        bev_features = torch.max(lidar_features, dim=2)[0]
        return bev_features

    def forward(self, point_cloud):
        # Step 1: Voxelize the raw point cloud
        voxel_grid = self.voxelize(point_cloud)

        # Step 2: Encode the voxelized point cloud using 3D CNN
        lidar_features = self.encoder(voxel_grid)

        # Step 3: Flatten along the z-axis to obtain BEV features
        bev_features = self.bev_flatten(lidar_features)

        return bev_features


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class TerrainHeads(nn.Module):
    def __init__(self, inC, outC):
        super(TerrainHeads, self).__init__()

        self.head_terrain = nn.Sequential(
            nn.Conv2d(inC, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, outC, kernel_size=1, padding=0)
        )
        self.head_frict = nn.Sequential(
            nn.Conv2d(inC, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, outC, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_terrain = self.head_terrain(x)
        friction = self.head_frict(x)
        out = {
            'terrain': x_terrain,
            'friction': friction
        }
        return out


class BEVFusion(LiftSplatShoot):
    def __init__(self, grid_conf, data_aug_conf, outC=1):
        super().__init__(grid_conf, data_aug_conf)

        self.lidar_bev = LiDARBEV(grid_conf=grid_conf)
        self.bevencode = BevEncode(inC=128, outC=128)
        self.terrain_heads = TerrainHeads(inC=128, outC=outC)

    def forward(self, img_inputs, cloud_input):
        # Get BEV features from camera inputs
        cam_feat_bev = self.get_voxels(*img_inputs)  # Shape (B, D, H, W)

        # Get BEV features from LiDAR inputs
        lidar_feat_bev = self.lidar_bev(cloud_input)  # Shape (B, D, H, W)

        # Concatenate the two BEV features
        feat_bev = torch.cat([cam_feat_bev, lidar_feat_bev], dim=1)  # Shape (B, 2xD, H, W)

        # Encode the concatenated BEV features
        fused_bev_feat = self.bevencode(feat_bev)  # Shape (B, 2xD, H, W)

        # Terrain head
        out = self.terrain_heads(fused_bev_feat)  # Dict, values of shape (B, 1, H, W)

        return out
