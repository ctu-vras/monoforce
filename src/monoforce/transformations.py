from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .cloudproc import position


__all__ = [
    'transform_cloud',
    'xyz_rpy_to_matrix',
    'rot2rpy',
    'rpy2rot',
]

def transform_cloud(cloud, Tr):
    if cloud.dtype.names is not None:
        points = position(cloud)
        points = transform_cloud(points, Tr)
        cloud = cloud.copy()
        cloud['x'] = points[:, 0]
        cloud['y'] = points[:, 1]
        cloud['z'] = points[:, 2]
        return cloud
    assert cloud.ndim == 2
    assert cloud.shape[1] == 3  # (N, 3)
    cloud_tr = np.matmul(Tr[:3, :3], cloud.T) + Tr[:3, 3:]
    return cloud_tr.T

def xyz_rpy_to_matrix(xyz_rpy):
    t = xyz_rpy[:3]
    R = Rotation.from_euler('xyz', xyz_rpy[3:]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def rot2rpy(R):
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
                       [0, torch.sin(roll), torch.cos(roll)]])

    RY = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)],
                       [0, 1, 0],
                       [-torch.sin(pitch), 0, torch.cos(pitch)]])

    RZ = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0],
                       [torch.sin(yaw), torch.cos(yaw), 0],
                       [0, 0, 1]])
    return RZ @ RY @ RX

def test_rpy():
    from tqdm import tqdm
    for _ in tqdm(range(10)):
        R = torch.as_tensor(Rotation.random().as_matrix())
        roll, pitch, yaw = rot2rpy(R)
        R2 = rpy2rot(roll, pitch, yaw)
        assert torch.allclose(R, R2)

        roll2, pitch2, yaw2 = torch.as_tensor(Rotation.from_matrix(R2).as_euler('xyz'))
        assert torch.allclose(roll, roll2)
        assert torch.allclose(pitch, pitch2)
        assert torch.allclose(yaw, yaw2)
        print(yaw.item() / np.pi * 180, yaw2.item() / np.pi * 180)


def test_transform_cloud():
    import open3d as o3d
    from .datasets.data import HMTrajData
    from numpy.lib.recfunctions import structured_to_unstructured as stu

    # Load traversability data
    path = '/home/ruslan/data/bags/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'
    ds = HMTrajData(path)
    print('Dataset contains %i samples' % len(ds))
    # Choose data sample
    # i = np.random.choice(range(len(ds)))
    i = 0
    cloud, traj, _ = ds[i]
    points = stu(cloud[['x', 'y', 'z']])
    print('%i-th sample contains point cloud of shape: %s' % (i, points.shape))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.0, 0.0, 1.0])

    pcd2 = o3d.geometry.PointCloud()
    Tr = np.eye(4)
    Tr[:3, :3] = Rotation.from_euler('z', angles=45, degrees=True).as_matrix()
    Tr[:3, 3] = [0.0, 0.0, 1.0]
    pcd2.points = o3d.utility.Vector3dVector(transform_cloud(points, Tr))
    pcd2.paint_uniform_color([1.0, 0.0, 0.0])

    o3d.visualization.draw_geometries([pcd, pcd2])


if __name__ == "__main__":
    # test_transform_cloud()
    test_rpy()

