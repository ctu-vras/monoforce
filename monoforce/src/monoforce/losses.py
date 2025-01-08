import torch


__all__ = [
    'rotation_difference',
    'translation_difference',
    'total_variation',
    'hm_loss',
    'slerp',
    'physics_loss'
]


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
    tr = dR.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    cos = (tr - 1) / 2.
    cos = torch.clip(cos, min=-1, max=1.)
    theta = torch.arccos(cos)
    theta = theta ** 2
    if reduction == 'mean':
        return theta.mean()
    elif reduction == 'sum':
        return theta.sum()
    else:
        return theta


def total_variation(heightmap):
    h, w = heightmap.shape[-2:]
    # Compute the total variation of a heightmap
    tv = torch.sum(torch.abs(heightmap[..., :, :-1] - heightmap[..., :, 1:])) + \
         torch.sum(torch.abs(heightmap[..., :-1, :] - heightmap[..., 1:, :]))
    tv = tv / (h * w)
    return tv


def hm_loss(height_pred, height_gt, weights=None, h_max=None):
    assert height_pred.shape == height_gt.shape, 'Height prediction and ground truth must have the same shape'
    if weights is None:
        weights = torch.ones_like(height_gt)
    assert weights.shape == height_gt.shape, 'Weights and height ground truth must have the same shape'

    if h_max is not None:
        # limit heightmap values to the physical limits: [-h_max, h_max]
        limit_fn = lambda x: h_max * torch.tanh(x)
        height_pred = limit_fn(height_pred)

    # remove nan values if any
    mask_valid = ~(torch.isnan(height_pred) | torch.isnan(height_gt))
    height_gt = height_gt[mask_valid]
    height_pred = height_pred[mask_valid]
    weights = weights[mask_valid]

    # compute weighted loss
    pred = height_pred * weights
    gt = height_gt * weights
    loss = ((pred - gt) ** 2).mean()

    return loss


def physics_loss(states_pred, states_gt, pred_ts, gt_ts, gamma=0.9, rotation_loss=False):
    """
    Compute the physics loss between predicted and ground truth states.
    :param states_pred: predicted states [N x T1 x 3]
    :param states_gt: ground truth states [N x T2 x 3]
    :param pred_ts: predicted timestamps N x T1
    :param gt_ts: ground truth timestamps N x T2
    :param gamma: time weight discount factor, w = 1 / (1 + gamma * t).
    """
    # unpack states
    X = states_gt[0]
    X_pred = states_pred[0]

    # find the closest timesteps in the trajectory to the ground truth timesteps
    ts_ids = torch.argmin(torch.abs(pred_ts.unsqueeze(1) - gt_ts.unsqueeze(2)), dim=2)

    # get the predicted states at the closest timesteps to the ground truth timesteps
    X_pred_gt_ts = X_pred[torch.arange(X.shape[0]).unsqueeze(1), ts_ids]

    # compute time weights: farthest timesteps have the least weight, w = 1 / (1 + t)
    time_weights = 1. / (1. + gamma * gt_ts.unsqueeze(2))
    pred = X_pred_gt_ts * time_weights
    gt = X * time_weights

    # trajectory position (xyz) MSE loss
    loss = ((pred - gt) ** 2).mean()

    if rotation_loss:
        # add trajectories rotation difference to loss
        R = states_gt[2]
        R_pred_gt_ts = states_pred[2][torch.arange(R.shape[0]).unsqueeze(1), ts_ids]
        loss_rot = rotation_difference(R_pred_gt_ts, R, reduction='none')
        loss_rot = (loss_rot * time_weights).mean()

        return loss, loss_rot

    return loss
