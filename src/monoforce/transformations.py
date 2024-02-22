from __future__ import absolute_import, division, print_function
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .utils import position
import numpy.matlib as matlib
from math import sin, cos, atan2, sqrt


__all__ = [
    'transform_cloud',
    'xyz_rpy_to_matrix',
    'rot2rpy',
    'rpy2rot',
]

MATRIX_MATCH_TOLERANCE = 1e-4


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

def build_se3_transform(xyzrpy):
    """Creates an SE3 transform from translation and Euler angles.

    Args:
        xyzrpy (list[float]): translation and Euler angles for transform. Must have six components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: SE3 homogeneous transformation matrix

    Raises:
        ValueError: if `len(xyzrpy) != 6`

    """
    if len(xyzrpy) != 6:
        raise ValueError("Must supply 6 values to build transform")

    se3 = matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3


def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.

    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix

    Raises:
        ValueError: if `len(rpy) != 3`.

    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    R_x = np.matrix([[1, 0, 0],
                     [0, cos(rpy[0]), -sin(rpy[0])],
                     [0, sin(rpy[0]), cos(rpy[0])]])
    R_y = np.matrix([[cos(rpy[1]), 0, sin(rpy[1])],
                     [0, 1, 0],
                     [-sin(rpy[1]), 0, cos(rpy[1])]])
    R_z = np.matrix([[cos(rpy[2]), -sin(rpy[2]), 0],
                     [sin(rpy[2]), cos(rpy[2]), 0],
                     [0, 0, 1]])
    R_zyx = R_z * R_y * R_x
    return R_zyx


def so3_to_euler(so3):
    """Converts an SO3 rotation matrix to Euler angles

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.matrixlib.defmatrix.matrix: list of Euler angles (size 3)

    Raises:
        ValueError: if so3 is not 3x3
        ValueError: if a valid Euler parametrisation cannot be found

    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")
    roll = atan2(so3[2, 1], so3[2, 2])
    yaw = atan2(so3[1, 0], so3[0, 0])
    denom = sqrt(so3[0, 0] ** 2 + so3[1, 0] ** 2)
    pitch_poss = [atan2(-so3[2, 0], denom), atan2(-so3[2, 0], -denom)]

    R = euler_to_so3((roll, pitch_poss[0], yaw))

    if (so3 - R).sum() < MATRIX_MATCH_TOLERANCE:
        return np.matrix([roll, pitch_poss[0], yaw])
    else:
        R = euler_to_so3((roll, pitch_poss[1], yaw))
        if (so3 - R).sum() > MATRIX_MATCH_TOLERANCE:
            raise ValueError("Could not find valid pitch angle")
        return np.matrix([roll, pitch_poss[1], yaw])


def so3_to_quaternion(so3):
    """Converts an SO3 rotation matrix to a quaternion

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.ndarray: quaternion [w, x, y, z]

    Raises:
        ValueError: if so3 is not 3x3
    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")

    R_xx = so3[0, 0]
    R_xy = so3[0, 1]
    R_xz = so3[0, 2]
    R_yx = so3[1, 0]
    R_yy = so3[1, 1]
    R_yz = so3[1, 2]
    R_zx = so3[2, 0]
    R_zy = so3[2, 1]
    R_zz = so3[2, 2]

    try:
        w = sqrt(so3.trace() + 1) / 2
    except(ValueError):
        # w is non-real
        w = 0

    # Due to numerical precision the value passed to `sqrt` may be a negative of the order 1e-15.
    # To avoid this error we clip these values to a minimum value of 0.
    x = sqrt(max(1 + R_xx - R_yy - R_zz, 0)) / 2
    y = sqrt(max(1 + R_yy - R_xx - R_zz, 0)) / 2
    z = sqrt(max(1 + R_zz - R_yy - R_xx, 0)) / 2

    max_index = max(range(4), key=[w, x, y, z].__getitem__)

    if max_index == 0:
        x = (R_zy - R_yz) / (4 * w)
        y = (R_xz - R_zx) / (4 * w)
        z = (R_yx - R_xy) / (4 * w)
    elif max_index == 1:
        w = (R_zy - R_yz) / (4 * x)
        y = (R_xy + R_yx) / (4 * x)
        z = (R_zx + R_xz) / (4 * x)
    elif max_index == 2:
        w = (R_xz - R_zx) / (4 * y)
        x = (R_xy + R_yx) / (4 * y)
        z = (R_yz + R_zy) / (4 * y)
    elif max_index == 3:
        w = (R_yx - R_xy) / (4 * z)
        x = (R_zx + R_xz) / (4 * z)
        y = (R_yz + R_zy) / (4 * z)

    return np.array([w, x, y, z])


def se3_to_components(se3):
    """Converts an SE3 rotation matrix to linear translation and Euler angles

    Args:
        se3: 4x4 transformation matrix

    Returns:
        numpy.matrixlib.defmatrix.matrix: list of [x, y, z, roll, pitch, yaw]

    Raises:
        ValueError: if se3 is not 4x4
        ValueError: if a valid Euler parametrisation cannot be found

    """
    if se3.shape != (4, 4):
        raise ValueError("SE3 transform must be a 4x4 matrix")
    xyzrpy = np.empty(6)
    xyzrpy[0:3] = se3[0:3, 3].transpose()
    xyzrpy[3:6] = so3_to_euler(se3[0:3, 0:3])
    return xyzrpy

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
    from .datasets.data import DEMPathData
    from numpy.lib.recfunctions import structured_to_unstructured as stu

    # Load traversability data
    path = '/home/ruslan/data/bags/robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34_trav/'
    ds = DEMPathData(path)
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

