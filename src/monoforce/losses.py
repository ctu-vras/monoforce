import torch
import numpy as np
from .models import RigidBodySoftTerrain, State
from .config import DPhysConfig
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.ops.knn import knn_points
from scipy.spatial.transform import Rotation


__all__ = [
    'traj_dist',
    'sampled_traj_dist',
    'sampled_traj_dist_with_interpolation',
    'goal_point_loss',
    'control_loss',
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


def slerp(q1, q2, t, diff_thresh=0.9995):
    assert isinstance(q1, torch.Tensor) and isinstance(q2, torch.Tensor)
    assert q1.shape == q2.shape == (4,)
    assert isinstance(t, torch.Tensor)
    # https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp
    #
    # q1 = [w1, x1, y1, z1]
    # q2 = [w2, x2, y2, z2]
    # q3 = [w3, x3, y3, z3]

    # dot product
    dot = (q1 * q2).sum()
    # if q1 and q2 are close, use linear interpolation
    if dot > diff_thresh:
        q3 = (q1[:, None] + t * (q2 - q1)[:, None]).T
        return q3 / torch.norm(q3)
    # if q1 and q2 are not close, use spherical interpolation
    theta_0 = torch.acos(dot)
    theta = theta_0 * t
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

def interpolate_rotations(R_from, R_to, t, normalize_time=True):
    assert isinstance(R_from, torch.Tensor) and isinstance(R_to, torch.Tensor)
    assert R_from.shape == R_to.shape == (3, 3)
    assert isinstance(t, torch.Tensor)
    if normalize_time:
        t = t - t.min()
        t = t / t.max()
    q1 = matrix_to_quaternion(R_from)
    q2 = matrix_to_quaternion(R_to)
    q3 = slerp(q1, q2, t)
    return quaternion_to_matrix(q3)


def translation_difference(x1, x2, reduction='mean'):
    assert isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor)
    assert x1.ndim == x2.ndim == 3
    N = x1.shape[0]
    assert N > 0
    assert x1.shape == x2.shape == (N, 3, 1)  # N x 3 x 1
    if reduction == 'mean':
        return torch.norm(x1 - x2, dim=1).mean()
    elif reduction == 'sum':
        return torch.norm(x1 - x2, dim=1).sum()
    else:
        return torch.norm(x1 - x2, dim=1)


def rotation_difference(R1, R2, reduction='mean'):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/#:~:text=The%20difference%20rotation%20matrix%20that,matrix%20R%20%3D%20P%20Q%20%E2%88%97%20.
    assert isinstance(R1, torch.Tensor) and isinstance(R2, torch.Tensor)
    assert R1.ndim == R2.ndim == 3  # N x 3 x 3
    N = R1.shape[0]
    assert N > 0
    assert R1.shape == R2.shape == (N, 3, 3)
    dR = R1 @ R2.transpose(dim0=1, dim1=2)
    # trace
    tr = dR.diagonal(dim1=1, dim2=2).sum(dim=1)
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


def sampled_traj_dist(system, states_true, tt, tt_true=None, norm_loss=False,
                      trans_cost_weight=1., rot_cost_weight=1.):
    assert isinstance(system, RigidBodySoftTerrain)
    states_true = tuple([x.detach() for x in states_true])

    xyz_true, R_true = states_true[:2]
    xyz_true = xyz_true.detach()
    R_true = R_true.detach()

    if tt_true is None:
        # find indices of x that are closest to x_true
        t0, s0 = system.get_initial_state()
        states = system.sim(s0, tt)
        xyz = states[0]
        ids = find_dist_correspondences(xyz1=xyz.squeeze(2), xyz2=xyz_true.squeeze(2))
    else:
        # find indices of tt that are closest to tt_true
        ids = find_time_correspondences(tt1=tt, tt2=tt_true)

    loss_total = torch.as_tensor(0.).to(system.device)
    for i in range(len(xyz_true) - 1):
        st0 = tuple([s[i].detach() for s in states_true])

        t_interval = tt[ids[i]:ids[i+1]]
        states = system.sim(st0, t_interval)

        # loss computed only for the last state of the predicted trajectory
        loss_tran = translation_difference(x1=states[0][-1][None], x2=xyz_true[i+1][None])
        loss_rot = rotation_difference(R1=states[1][-1][None], R2=R_true[i+1][None])

        loss = trans_cost_weight * loss_tran + rot_cost_weight * loss_rot
        loss_total += loss.clone()

    if norm_loss:
        loss_total = loss_total / (len(xyz_true) - 1)

    return loss_total


def sampled_traj_dist_with_interpolation(system, states_true, tt, tt_true, cfg):
    """
    Interpolates true states to match the time stamps of the predicted states.
    Then computes the loss between the predicted and interpolated true states.

    :param system: RigidBodySoftTerrain, differentiable physics model
    :param states_true: tuple of tensors (xyz, R, v, omega, forces)
    :param tt: time stamps of the predicted states
    :param tt_true: time stamps of the true states
    :param cfg: Config, parameters
    :return: loss, L2 loss between the predicted and interpolated true states (translation and rotation parts)
    """
    st0 = tuple([x[0] for x in states_true])
    states = system.sim(st0, tt)
    states_true_interp = states.copy()

    xyz_true, R_true, vel_true, omega_true, forces_true = states_true
    # interpolate true poses so that they are the same length as the predicted states (or their time stamps)
    # normalize time stamps to [0, 1]
    tt = tt - tt.min()
    tt = tt / tt.max()
    tt_true = tt_true - tt_true.min()
    tt_true = tt_true / tt_true.max()
    ids = find_time_correspondences(tt, tt_true)

    xyz_true_interp = []
    R_true_interp = []
    vel_true_interp = []
    omega_true_interp = []
    for i in range(len(ids) - 1):
        # print('Interpolating true states for time moments: %.2f .. %.2f' % (t_interval.min(), t_interval.max()))
        n = ids[i+1] - ids[i]

        # interpolate positions, velocities, angular velocities for time interval t_interval
        t_interp = torch.linspace(0, 1, n).to(system.device)
        xyz_true_interval = linear_interpolation(xyz_true[i, :], xyz_true[i + 1, :], t_interp)
        vel_true_interval = linear_interpolation(vel_true[i, :], vel_true[i + 1, :], t_interp)
        omega_true_interval = linear_interpolation(omega_true[i, :], omega_true[i + 1, :], t_interp)
        # interpolate rotation matrices R_true for time interval t_interval
        R_true_interval = interpolate_rotations(R_true[i], R_true[i + 1], t_interp)  # (n x 3 x 3)

        xyz_true_interp.append(xyz_true_interval)
        R_true_interp.append(R_true_interval)
        vel_true_interp.append(vel_true_interval)
        omega_true_interp.append(omega_true_interval)
    xyz_true_interp += [xyz_true[-1].view(1, 3, 1)]
    R_true_interp += [R_true[-1].view(1, 3, 3)]
    vel_true_interp += [vel_true[-1].view(1, 3, 1)]
    omega_true_interp += [omega_true[-1].view(1, 3, 1)]

    xyz_true_interp = torch.concat(xyz_true_interp, dim=0)  # (n x 3 x 1)
    R_true_interp = torch.concat(R_true_interp, dim=0)  # (n x 3 x 3)
    vel_true_interp = torch.concat(vel_true_interp, dim=0)  # (n x 3 x 1)
    omega_true_interp = torch.concat(omega_true_interp, dim=0)  # (n x 3 x 1)

    # # visualize interpolated true states
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # # plot xyz_true_interp as 3d trajectory
    # ax.plot(xyz_true[:, 0, 0].detach().cpu().numpy(),
    #         xyz_true[:, 1, 0].detach().cpu().numpy(),
    #         xyz_true[:, 2, 0].detach().cpu().numpy(), 'ro', markersize=5)
    # # plot xyz_true_interp as 3d trajectory
    # ax.plot(xyz_true_interp[:, 0, 0].detach().cpu().numpy(),
    #         xyz_true_interp[:, 1, 0].detach().cpu().numpy(),
    #         xyz_true_interp[:, 2, 0].detach().cpu().numpy(), 'b')
    # # plot xyz predicted states as 3d trajectory
    # ax.plot(states[0][:, 0, 0].detach().cpu().numpy(),
    #         states[0][:, 1, 0].detach().cpu().numpy(),
    #         states[0][:, 2, 0].detach().cpu().numpy(), 'r')
    # set_axes_equal(ax)
    # plt.legend(['xyz_true', 'xyz_true_interp', 'xyz_pred'])
    # plt.show()

    states_true_interp[0] = xyz_true_interp
    states_true_interp[1] = R_true_interp
    states_true_interp[2] = vel_true_interp
    states_true_interp[3] = omega_true_interp

    assert len(tt) == len(states_true_interp[0])
    states_true_interp = tuple([x.detach() for x in states_true_interp])

    loss = torch.tensor(0.).to(cfg.device)
    for tm in range(int(cfg.n_samples / cfg.sample_len)):
        st0 = (states_true_interp[0][tm * cfg.sample_len].detach(), states_true_interp[1][tm * cfg.sample_len].detach(),
               states_true_interp[2][tm * cfg.sample_len].detach(), states_true_interp[3][tm * cfg.sample_len].detach(),
               states_true_interp[4][tm * cfg.sample_len].detach())

        states = system.sim(st0, tt[tm * cfg.sample_len:(tm + 1) * cfg.sample_len])
        loss_tran = translation_difference(x1=states[0],
                                           x2=states_true_interp[0][tm * cfg.sample_len:(tm + 1) * cfg.sample_len])
        loss_rot = rotation_difference(R1=states[1], R2=states_true_interp[1][tm * cfg.sample_len:(tm + 1) * cfg.sample_len])
        loss += loss_tran + loss_rot

    return loss


def goal_point_loss(system, states_true, tt, tt_true=None, trans_weight=0.5, rot_weight=0.5):
    assert isinstance(system, RigidBodySoftTerrain)

    st0 = tuple([s[0].detach() for s in states_true])
    states = system.sim(st0, tt)

    x_true_xyz, x_true_R = states_true[:2]
    x_true_xyz = x_true_xyz.detach()
    x_true_R = x_true_R.detach()

    loss_tran = translation_difference(x1=states[0][-1:], x2=x_true_xyz[-1:])
    loss_rot = rotation_difference(R1=states[1][-1:], R2=x_true_R[-1:])
    loss = trans_weight * loss_tran + rot_weight * loss_rot

    return loss


def control_loss(system, states_true, tt, cfg, return_states=False, log=False):
    n_true_states = len(states_true[0])
    state = system.state
    states = []

    loss_trans_sum = torch.tensor(0., device=cfg.device)
    loss_rot_sum = torch.tensor(0., device=cfg.device)
    for i in range(n_true_states - 1):
        # print('Going from pose %s -> to waypoint %s' % (state[0].squeeze(), xyz_true[i + 1].squeeze()))
        time_interval = tt[i * cfg.n_samples // (n_true_states - 1):(i + 1) * cfg.n_samples // (n_true_states - 1)]
        # states_interval = system.sim(state, time_interval)

        goal_state = (states_true[0][i + 1].view(3, 1),
                      states_true[1][i + 1].view(3, 3),
                      states_true[2][i + 1].view(3, 1),
                      states_true[3][i + 1].view(3, 1),
                      states_true[4][i + 1])
        states_interval = system.sim_control(state, goal_state, time_interval)

        pos_x, pos_R, vel_x, vel_omega, forces = states_interval
        # update state
        state = State(pos_x[-1].view(3, 1),
                      pos_R[-1].view(3, 3),
                      vel_x[-1].view(3, 1),
                      vel_omega[-1].view(3, 1),
                      forces[-1])

        # compute loss
        loss_xyz = translation_difference(state[0].view(1, 3, 1), states_true[0][i + 1].view(1, 3, 1))
        loss_rpy = rotation_difference(state[1].view(1, 3, 3), states_true[1][i + 1].view(1, 3, 3))
        loss_trans_sum += loss_xyz
        loss_rot_sum += loss_rpy
        states.append(states_interval)

    loss_trans_sum /= n_true_states
    loss_rot_sum /= n_true_states
    loss = loss_trans_sum + loss_rot_sum

    if log:
        print('Loss: %.3f (trans: %.3f [m], rot: %.3f [rad])' %
              (loss.item(), loss_trans_sum.item(), loss_rot_sum.item()))

    if return_states:
        pos_x = torch.cat([x[0] for x in states], dim=0)
        pos_R = torch.cat([x[1] for x in states], dim=0)
        vel_x = torch.cat([x[2] for x in states], dim=0)
        vel_omega = torch.cat([x[3] for x in states], dim=0)
        forces = torch.cat([x[4] for x in states], dim=0)
        states = (pos_x, pos_R, vel_x, vel_omega, forces)
        return loss, states
    return loss


def test_loss_functions():
    from matplotlib import pyplot as plt

    system_true = RigidBodySoftTerrain(height=np.zeros((10, 10)),
                                       grid_res=1.,
                                       damping=10.0, elasticity=10.0, friction=1., mass=10.0,
                                       state=State(xyz=np.asarray([0.0, 0.0, 1.0]),
                                                   rot=Rotation.from_euler('xyz', np.array([0.0, 0.0, 0.])).as_matrix()),
                                       vel_tracks=np.array([2., 2.]),
                                       use_ode=False)

    total_time = 3.
    tt_true = torch.linspace(0., total_time, int(total_time) * 100)
    states_true = system_true.sim(system_true.state, tt_true)

    tt_true = tt_true[::10]
    states_true = tuple([s[::10] for s in states_true])

    # losses = [full_traj_loss, sampled_traj_loss, goal_point_loss]
    losses = [sampled_traj_dist]

    for loss_fn in losses:
        for v_noise in np.linspace(0., 0.1, 5):
            system = RigidBodySoftTerrain(height=np.zeros((10, 10)),
                                          grid_res=1.,
                                          damping=10.0, elasticity=10.0, friction=1., mass=10.0,
                                          state=State(xyz=np.asarray([0.0, 0.0, 1.0]),
                                                      rot=Rotation.from_euler('xyz', np.array([0.0, 0.0, 0.2])).as_matrix()),
                                          vel_tracks=np.array([2., 2.]) + np.array([0, 1]) * v_noise,
                                          use_ode=False)

            tt = torch.linspace(0., total_time, int(total_time) * 100)
            st0 = tuple([s[0].detach() for s in states_true])
            states = system.sim(st0, tt)
            system.update_trajectory(tt)

            loss = loss_fn(system, states_true, tt, tt_true)
            print(loss_fn.__name__, ': ', loss.item(), 'v_noise: ', v_noise)

            plt.figure()
            plt.axis('equal')
            plt.plot(states_true[0][:, 0], states_true[0][:, 1], 'r.-')
            plt.plot(states[0][:, 0], states[0][:, 1], 'b.-')
            plt.show()


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

def test_closest_points():
    from scipy.spatial.transform import Rotation
    from matplotlib import pyplot as plt
    from .datasets.utils import get_robingas_data, get_kkt_data
    from .vis import set_axes_equal

    cfg = DPhysConfig()
    cfg.traj_sim_time = 3.

    # pts1 = np.random.uniform(-1, 1, (10, 3))
    # R = Rotation.from_euler('xyz', (0, 0, 30), degrees=True).as_matrix()
    # t = np.array([0.1, 0.2, 0.3])
    # pts2 = np.matmul(R, pts1.T).T + t

    height, traj = get_robingas_data(cfg)
    # height = np.zeros_like(height)
    poses = traj['poses']
    vels = traj['vels']
    omegas = traj['omegas']
    t_stamps = traj['stamps']
    tt_true = torch.as_tensor(t_stamps)

    system = RigidBodySoftTerrain(height=height['z'],
                                  grid_res=0.4,
                                  damping=10.0, elasticity=10.0, friction=1., mass=10.0,
                                  state=State(xyz=poses[0][:3, 3] + np.asarray([0.0, 0.0, 1.0]),
                                              rot=Rotation.from_matrix(poses[0][:3, :3]),
                                              vel=vels[0][:3],
                                              omega=omegas[0][:3]),
                                  vel_tracks=np.array([2.5, 2.5]),
                                  use_ode=False)

    tt = torch.linspace(0., cfg.traj_sim_time, int(cfg.traj_sim_time) * 100)
    t0, st0 = system.get_initial_state()
    states = system.sim(st0, tt)

    pts1 = states[0][:, :3].detach().numpy().squeeze()
    pts2 = poses[:, :3, 3]

    pts1 = torch.as_tensor(pts1)
    pts2 = torch.as_tensor(pts2)

    # ids = find_time_correspondences(tt, tt_true)
    ids = find_dist_correspondences(pts1, pts2)

    # draw point clouds in 3D
    plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')
    ax.scatter(pts1[:, 0], pts1[:, 1], pts1[:, 2], c='r')
    ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], c='b')
    # draw lines between closest points
    for i in range(pts2.shape[0]):
        ax.plot([pts2[i, 0], pts1[ids[i], 0]], [pts2[i, 1], pts1[ids[i], 1]], [pts2[i, 2], pts1[ids[i], 2]], c='g')
        # set distances as text labels
        # ax.text(pts2[i, 0], pts2[i, 1], pts2[i, 2], '%.2f' % dists[i], size=10, zorder=1, color='k')
    set_axes_equal(ax)
    plt.show()


if __name__ == '__main__':
    with torch.no_grad():
        test_loss_functions()
    test_slerp()
    for _ in range(5):
        test_closest_points()
