import torch
import numpy as np
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.ops.knn import knn_points


__all__ = [
    'traj_dist',
    'rotation_difference',
    'translation_difference',
    'find_dist_correspondences',
    'find_time_correspondences',
    'total_variation',
    'RMSE',
]


class RMSE:
    def __init__(self, reduction='mean', eps=1e-6):
        assert reduction in ['mean', 'sum', 'none']
        assert isinstance(eps, float) and eps > 0
        self.reduction = reduction
        self.eps = eps

    def __call__(self, x, x_true):
        assert isinstance(x, torch.Tensor) and isinstance(x_true, torch.Tensor)
        assert x.shape == x_true.shape
        loss = torch.sqrt((x - x_true) ** 2 + self.eps)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


def slerp(q1, q2, t_interval, diff_thresh=0.9995):
    assert isinstance(q1, torch.Tensor) and isinstance(q2, torch.Tensor)
    assert q1.shape == q2.shape == (4,)
    assert isinstance(t_interval, torch.Tensor)
    # https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp

    # dot product
    dot = (q1 * q2).sum()
    # if q1 and q2 are close, use linear interpolation
    if dot > diff_thresh:
        q3 = (q1[:, None] + t_interval * (q2 - q1)[:, None]).T
        return q3 / torch.norm(q3)
    # if q1 and q2 are not close, use spherical interpolation
    theta_0 = torch.acos(dot)
    theta = theta_0 * t_interval
    sin_theta = torch.sin(theta)
    sin_theta_0 = torch.sin(theta_0)
    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q3 = s0[:, None] * q1 + s1[:, None] * q2
    return q3 / torch.norm(q3, dim=1, keepdim=True)

def linear_interpolation(xyz_from, xyz_to, t_interval):
    assert isinstance(xyz_from, torch.Tensor) and isinstance(xyz_to, torch.Tensor)
    assert xyz_from.shape == xyz_to.shape == (3, 1)
    N = len(t_interval)
    xyz_interp = torch.lerp(xyz_from, xyz_to, t_interval).T
    xyz_interp = xyz_interp.reshape(N, 3, 1)
    return xyz_interp

def interpolate_rotations(R_from, R_to, t_interval, normalize_time=True):
    assert isinstance(R_from, torch.Tensor) and isinstance(R_to, torch.Tensor)
    assert R_from.shape == R_to.shape == (3, 3)
    assert isinstance(t_interval, torch.Tensor)
    if normalize_time:
        t_interval = t_interval - t_interval.min()
        t_interval = t_interval / t_interval.max()
    q1 = matrix_to_quaternion(R_from)
    q2 = matrix_to_quaternion(R_to)
    q3 = slerp(q1, q2, t_interval)
    return quaternion_to_matrix(q3)


def translation_difference(x1, x2, reduction='mean'):
    assert isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor)
    assert x1.shape == x2.shape
    assert x1.shape[-1] == 3
    if reduction == 'mean':
        return torch.norm(x1 - x2, dim=-1).mean()
    elif reduction == 'sum':
        return torch.norm(x1 - x2, dim=-1).sum()
    else:
        return torch.norm(x1 - x2, dim=-1)


def rotation_difference(R1, R2, reduction='mean'):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/#:~:text=The%20difference%20rotation%20matrix%20that,matrix%20R%20%3D%20P%20Q%20%E2%88%97%20.
    assert isinstance(R1, torch.Tensor) and isinstance(R2, torch.Tensor)
    assert R1.shape == R2.shape # for example N x 3 x 3
    assert R1.shape[-2:] == (3, 3)
    dR = R1 @ R2.transpose(dim0=-2, dim1=-1)
    # trace
    tr = dR.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos = (tr - 1) / 2.
    cos = torch.clip(cos, min=-1, max=1.)
    theta = torch.arccos(cos)
    if reduction == 'mean':
        return theta.abs().mean()
    elif reduction == 'sum':
        return theta.abs().sum()
    else:
        return theta


def total_variation(heightmap):
    h, w = heightmap.shape[-2:]
    # Compute the total variation of the image
    tv = torch.sum(torch.abs(heightmap[..., :, :-1] - heightmap[..., :, 1:])) + \
         torch.sum(torch.abs(heightmap[..., :-1, :] - heightmap[..., 1:, :]))
    tv /= h * w
    return tv


def find_dist_correspondences(xyz1, xyz2, return_dist=False):
    assert isinstance(xyz1, torch.Tensor) and isinstance(xyz2, torch.Tensor)
    n1_pts = len(xyz1)
    n2_pts = len(xyz2)
    assert xyz1.shape == (n1_pts, 3)
    assert xyz2.shape == (n2_pts, 3)
    # returns indices of xyz1 that are closest to xyz2 in terms of euclidean distance
    xyz1 = torch.as_tensor(xyz1, dtype=torch.float32)
    xyz2 = torch.as_tensor(xyz2, dtype=torch.float32)
    dist, ids, _ = knn_points(xyz2[None], xyz1[None], K=1, norm=2)
    if return_dist:
        return ids.squeeze(), dist.squeeze()
    return ids.squeeze()

def find_time_correspondences(tt1, tt2):
    ids = torch.searchsorted(tt1, tt2)
    ids = torch.clamp(ids, 0, len(tt1) - 1)
    return ids


def traj_dist(states, states_true, tt=None, tt_true=None, return_trans_and_rot=False,
              trans_cost_weight=1., rot_cost_weight=1.):
    """
    Computes the distance between the predicted trajectory
    by the robot-terrain interaction system and the true trajectory.
    If time points are provided, the distance is computed at those time points.
    Otherwise, the distance is computed at the nearest points from the
    predicted trajectory to the true trajectory.

    @param states: predicted states
    @param states_true: ground truth states
    @param tt: time points of the predicted trajectory
    @param tt_true: time points of the true trajectory
    @param trans_cost_weight: weight of the translation distance
    @param rot_cost_weight: weight of the rotation distance
    @param return_trans_and_rot: if True, returns translation and rotation distances separately
    @return: weighted sum of translation and rotation distances
    """
    states_true = tuple([s.detach() for s in states_true])

    xyz = states[0]
    R = states[1]

    xyz_true, R_true = states_true[:2]
    xyz_true = xyz_true.detach()
    R_true = R_true.detach()

    if tt_true is None or tt is None:
        # find indices of x that are closest to x_true
        ids = find_dist_correspondences(xyz1=xyz.view(-1, 3), xyz2=xyz_true.view(-1, 3))
    else:
        # find indices of tt that are closest to tt_true
        ids = find_time_correspondences(tt1=tt, tt2=tt_true)

    loss_tran = translation_difference(x1=xyz[ids], x2=xyz_true)
    loss_rot = rotation_difference(R1=R[ids], R2=R_true)
    loss = trans_cost_weight * loss_tran + rot_cost_weight * loss_rot
    if return_trans_and_rot:
        return loss_tran, loss_rot

    return loss


def test_slerp():
    from scipy.spatial.transform import Rotation
    from scipy.spatial.transform import Slerp

    R1 = Rotation.from_euler('xyz', (0, 0, 0), degrees=True).as_matrix()
    R2 = Rotation.from_euler('xyz', (0, 0, 90), degrees=True).as_matrix()
    # R1 = Rotation.random().as_matrix()
    # R2 = Rotation.random().as_matrix()

    R1 = torch.as_tensor(R1)
    R2 = torch.as_tensor(R2)

    tt = torch.linspace(0.2, 2., 5)
    R_interp = interpolate_rotations(R1, R2, tt)

    # scipy slerp example
    rots = np.vstack([R1[None], R2[None]])
    rots = Rotation.from_matrix(rots)
    slerp = Slerp([tt.min(), tt.max()], rots)

    for t, R in zip(tt, R_interp):
        rot = Rotation.from_matrix(R.numpy())
        rot_scipy = slerp(t)
        yaw = rot.as_euler('xyz', degrees=True)[2]
        yaw_scipy = rot_scipy.as_euler('xyz', degrees=True)[2]
        print(t, yaw, yaw_scipy)
        assert np.allclose(yaw, yaw_scipy), 'slerp failed'
