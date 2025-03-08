import torch
import torch.nn as nn
from .voxelnet import LidarNet
from .lss import LiftSplatShoot, BevEncode


class BEVFusion(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC=1):
        super().__init__()
        self.lss = LiftSplatShoot(grid_conf, data_aug_conf)
        self.lidar_net = LidarNet(grid_conf=grid_conf, out_channels=64)
        self.bevencode = BevEncode(inC=2*64, outC=outC)

    def forward(self, img_inputs, cloud_input):
        # Get BEV features from camera inputs
        cam_feat_bev = self.lss.get_voxels(*img_inputs)  # Shape (B, Z, X, Y)

        # Get BEV features from LiDAR inputs
        lidar_feat_bev = self.lidar_net(cloud_input)  # Shape (B, Z, X, Y)

        # Concatenate the two BEV features
        feat_bev = torch.cat([cam_feat_bev, lidar_feat_bev], dim=1)  # Shape (B, 2xZ, X, Y)

        # Encode the concatenated BEV features
        out = self.bevencode(feat_bev)  # Shape (B, 1, X, Y)

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
