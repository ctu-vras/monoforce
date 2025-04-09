from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.spatial.transform import Rotation


__all__ = [
    'transform_cloud',
    'xyz_rpy_to_matrix',
    'rot2rpy',
    'rpy2rot',
    'pose_to_xyz_q',
]


def position(cloud):
    """Cloud to point positions (xyz)."""
    if cloud.dtype.names:
        x = structured_to_unstructured(cloud[['x', 'y', 'z']])
    else:
        x = cloud
    return x


def transform_cloud(cloud, Tr):
    assert isinstance(cloud, np.ndarray) or isinstance(cloud, torch.Tensor), type(cloud)
    assert isinstance(Tr, np.ndarray) or isinstance(Tr, torch.Tensor), type(Tr)
    if isinstance(cloud, np.ndarray) and cloud.dtype.names is not None:
        points = position(cloud)
        points = transform_cloud(points, Tr)
        cloud = cloud.copy()
        cloud['x'] = points[:, 0]
        cloud['y'] = points[:, 1]
        cloud['z'] = points[:, 2]
        return cloud
    assert cloud.ndim == 2
    assert cloud.shape[1] == 3  # (N, 3)
    cloud_tr = Tr[:3, :3] @ cloud.T + Tr[:3, 3:]
    return cloud_tr.T

def xyz_rpy_to_matrix(xyz_rpy):
    t = xyz_rpy[:3]
    R = Rotation.from_euler('xyz', xyz_rpy[3:]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def rot2rpy(R):
    assert isinstance(R, torch.Tensor) or isinstance(R, np.ndarray)
    assert R.shape == (3, 3)
    if isinstance(R, np.ndarray):
        R = torch.as_tensor(R)
    roll = torch.atan2(R[2, 1], R[2, 2])
    pitch = torch.atan2(-R[2, 0], torch.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    yaw = torch.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw

def rpy2rot(roll, pitch, yaw):
    roll = torch.as_tensor(roll)
    pitch = torch.as_tensor(pitch)
    yaw = torch.as_tensor(yaw)
    RX = torch.tensor([[1, 0, 0],
                       [0, torch.cos(roll), -torch.sin(roll)],
                       [0, torch.sin(roll), torch.cos(roll)]], dtype=torch.float32)

    RY = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                       [0, 1, 0],
                       [-torch.sin(pitch), 0, torch.cos(pitch)]], dtype=torch.float32)

    RZ = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                       [torch.sin(yaw), torch.cos(yaw), 0],
                       [0, 0, 1]], dtype=torch.float32)
    return RZ @ RY @ RX


def pose_to_xyz_q(pose):
    assert isinstance(pose, np.ndarray) or isinstance(pose, torch.Tensor)
    assert pose.shape == (4, 4)
    if isinstance(pose, np.ndarray):
        pose = torch.as_tensor(pose)
    xyz = pose[:3, 3]
    quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
    quat = torch.as_tensor(quat)
    xyz_q = torch.cat([xyz, quat])
    return xyz_q
