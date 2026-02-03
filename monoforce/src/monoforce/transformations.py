from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.spatial.transform import Rotation

try:
    Rotation_from_matrix = Rotation.from_matrix

    def Rotation_as_matrix(rot, *args, **kwargs):
        return rot.as_matrix(*args, **kwargs)
except:
    Rotation_from_matrix = Rotation.from_dcm

    def Rotation_as_matrix(rot, *args, **kwargs):
        return rot.as_dcm(*args, **kwargs)

# OpenBLAS with Numpy 1.17.4 on arm64 causes incorrect results of transform_cloud(); switch to ATLAS BLAS library.
import platform
if platform.machine() == 'aarch64':
    np_version = tuple(map(int, np.version.short_version.split('.')))
    if np_version < (1, 21, 1):
        import os
        import sys
        blas_lib = os.path.realpath('/etc/alternatives/libblas.so.3-aarch64-linux-gnu')
        if 'openblas' in blas_lib and np.version.version:
            print('After installing ATLAS, call this command and select it:', file=sys.stderr)
            print('sudo update-alternatives --config libblas.so.3-aarch64-linux-gnu', file=sys.stderr)
            assert False, ('OpenBLAS is buggy on this OS and Numpy. '
                           'Install and use libatlas-base-dev or upgrade numpy to > 1.21.0.')


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

    # This is workaround for buggy older OpenBLAS on arm64
    if isinstance(cloud, np.ndarray):
        nans = np.where(np.isnan(cloud))
        cloud[nans] = 123456.789

    cloud_tr = Tr[:3, :3] @ cloud.T + Tr[:3, 3:]
    cloud_tr = cloud_tr.T

    if isinstance(cloud, np.ndarray):
        cloud[nans] = np.nan
        cloud_tr[nans] = np.nan

    return cloud_tr

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
    quat = Rotation_from_matrix(pose[:3, :3]).as_quat()
    quat = torch.as_tensor(quat)
    xyz_q = torch.cat([xyz, quat])
    return xyz_q
