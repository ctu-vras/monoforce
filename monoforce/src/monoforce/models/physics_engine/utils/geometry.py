import torch

__all__ = [
    "normalized",
    "skew_symmetric",
    "rot_X",
    "rot_Y",
    "rot_Z",
    "global_to_local",
    "local_to_global",
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "quaternion_conjugate",
    "rotate_vector_by_quaternion",
    "global_to_local_q",
    "local_to_global_q",
    "yaw_from_R",
    "planar_rot_from_R3",
    "planar_rot_from_q",
    "quaternion_to_yaw",
    "quaternion_to_pitch",
    "quaternion_to_roll",
    "points_in_oriented_box",
    "pointcloud_bounding_volume",
    "extract_top_plane_from_box",
    "euler_to_quaternion",
    "quaternion_to_euler",
    "rotation_matrix_to_euler_zyx",
    "pitch_from_R",
    "roll_from_R",
    "inverse_quaternion",
    "points_within_circle",
    "rodrigues_rotation_matrix",
    "bbox_limits_to_points",
    "q_to_R",
]


def rodrigues_rotation_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Computes the rotation matrix using Rodrigues' rotation formula.

    Parameters:
    - axis: Rotation axis (3D vector).
    - angle: Rotation angle in radians.

    Returns:
    - Rotation matrix.
    """
    axis = normalized(axis).view(-1, 3)
    K = skew_symmetric(axis)
    return torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)


def normalized(x, eps=1e-8):
    """
    Normalizes the input tensor.

    Parameters:
    - x: Input tensor.
    - eps: Small value to avoid division by zero.

    Returns:
    - Normalized tensor.
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    norm.clamp_(min=eps)
    return x / norm


def unit_quaternion(batch_size: int = 1, device: str | torch.device = "cpu"):
    """
    Returns a unit quaternion tensor of shape (batch_size, 4).

    Parameters:
    - batch_size: Number of quaternions to generate.
    - device: Device on which to create the tensor.

    Returns:
    - Unit quaternion tensor.
    """
    q = torch.zeros(batch_size, 4, device=device)
    q[:, 0] = 1.0
    return q


def skew_symmetric(v):
    """
    Returns the skew-symmetric matrix of a vector.

    Parameters:
    - v: Input vector.

    Returns:
    - Skew-symmetric matrix of the input vector.
    """
    U = torch.zeros(v.shape[0], 3, 3, device=v.device)
    U[:, 0, 1] = -v[:, 2]
    U[:, 0, 2] = v[:, 1]
    U[:, 1, 2] = -v[:, 0]
    U[:, 1, 0] = v[:, 2]
    U[:, 2, 0] = -v[:, 1]
    U[:, 2, 1] = v[:, 0]
    return U


def rot_X(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view(-1, 1)
    cos_ang = torch.cos(theta)
    sin_ang = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    return torch.stack(
        [
            torch.cat([ones, zeros, zeros], dim=-1),
            torch.cat([zeros, cos_ang, -sin_ang], dim=-1),
            torch.cat([zeros, sin_ang, cos_ang], dim=-1),
        ],
        dim=1,
    )  # Stack along new dimension to create (B, 3, 3)


def rot_Y(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view(-1, 1)
    cos_ang = torch.cos(theta)
    sin_ang = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    return torch.stack(
        [
            torch.cat([cos_ang, zeros, sin_ang], dim=-1),
            torch.cat([zeros, ones, zeros], dim=-1),
            torch.cat([-sin_ang, zeros, cos_ang], dim=-1),
        ],
        dim=1,
    )  # Stack along new dimension to create (B, 3, 3)


def rot_Z(theta: torch.Tensor) -> torch.Tensor:
    theta = theta.view(-1, 1)
    cos_ang = torch.cos(theta)
    sin_ang = torch.sin(theta)
    zeros = torch.zeros_like(theta)
    ones = torch.ones_like(theta)
    return torch.stack(
        [
            torch.cat([cos_ang, -sin_ang, zeros], dim=-1),
            torch.cat([sin_ang, cos_ang, zeros], dim=-1),
            torch.cat([zeros, zeros, ones], dim=-1),
        ],
        dim=1,
    )  # Stack along new dimension to create (B, 3, 3)


def global_to_local(t: torch.Tensor, R: torch.Tensor, points: torch.Tensor):
    """
    Transforms the global coordinates to the local coordinates.

    Parameters:
    - t: Translation vector.
    - R: Rotation matrix in the global frame.
    - points: Global coordinates.

    Returns:
    - Local coordinates.
    """
    if points.dim() != 3:  # if points are not batched
        points = points.unsqueeze(0)  # (1, N, D)
    B, N, D = points.shape
    t = t.reshape(B, 1, D)
    R = R.reshape(B, D, D)
    return torch.bmm(points - t, R)  # Correspods to transposed rotation matrix -> inverse


def local_to_global(t: torch.Tensor, R: torch.Tensor, points: torch.Tensor):
    """
    Transforms the global coordinates to the local coordinates.

    Parameters:
    - t: Translation vector.
    - R: Rotation matrix in global frame.
    - points: Global coordinates.

    Returns:
    - Local coordinates.
    """
    if points.dim() != 3:  # if points are not batched
        points = points.unsqueeze(0)  # (1, N, D)
    B, N, D = points.shape
    t = t.reshape(B, 1, D)
    R = R.reshape(B, D, D)
    return torch.bmm(points, R.transpose(1, 2)) + t  # corresponds to original rotation matrix


def quaternion_multiply(q: torch.Tensor, r: torch.Tensor):
    """
    Multiplies two quaternions.
    q, r: Tensors of shape [B, 4]
    Returns: Tensor of shape [B, 4]
    """
    # q = [w, x, y, z]
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def q_to_R(q: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix.
    q: Tensor of shape [B, 4]
    Returns: Tensor of shape [B, 3, 3]
    """
    w, x, y, z = q.unbind(-1)
    B = q.shape[0]

    # Precompute products
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    R = torch.zeros(B, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = ww + xx - yy - zz
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = ww - xx + yy - zz
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = ww - xx - yy + zz

    return R


def quaternion_to_rotation_matrix(q: torch.Tensor):
    """
    Converts a quaternion to a rotation matrix.
    q: Tensor of shape [B, 4]
    Returns: Tensor of shape [B, 3, 3]
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = q.shape[0]

    # Precompute products
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    R = torch.zeros(B, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = ww + xx - yy - zz
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = ww - xx + yy - zz
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = ww - xx - yy + zz

    return R


def quaternion_conjugate(q):
    """
    Compute the conjugate of a quaternion.
    q: Tensor of shape (N, 4)
    Returns: Tensor of shape (N, 4)
    """
    return q * torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)


def rotate_vector_by_quaternion(v, q):
    """
    Rotate vector(s) v by quaternion(s) q using the direct cross-product method.
    v: Tensor of shape (B,N or 1, 3)
    q: Tensor of shape (B,N or 1, 4), assumed to be normalized quaternion
    Returns: Tensor of shape (..., 3)
    """
    # Split quaternion into scalar (q_w) and vector parts (q_xyz)
    q_w = q[..., None, 0:1]  # shape (..., 1)
    q_xyz = q[..., None, 1:]  # shape (..., 3)
    # Compute intermediate cross products
    uv = torch.cross(q_xyz, v, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    # Apply the rotation formula
    return v + 2 * (q_w * uv + uuv)


def global_to_local_q(t: torch.Tensor, q: torch.Tensor, points: torch.Tensor):
    """
    Transforms the global coordinates to the local coordinates.

    Parameters:
    - t: Translation vector.
    - q: Rotation quaternion in the global frame.
    - points: Global coordinates.

    Returns:
    - Local coordinates.
    """
    if points.dim() != 3:  # if points are not batched
        points = points.unsqueeze(0)  # (1, N, D)
    B, N, D = points.shape
    t = t.reshape(B, 1, D)
    q = q.reshape(B, 4)
    return rotate_vector_by_quaternion(points - t, quaternion_conjugate(q))


def local_to_global_q(t: torch.Tensor, q: torch.Tensor, points: torch.Tensor):
    """
    Transforms the local coordinates to the global coordinates.

    Parameters:
    - t: Translation vector.
    - q: Rotation quaternion in global frame.
    - points: Global coordinates.

    Returns:
    - Global coordinates.
    """
    if points.dim() != 3:  # if points are not batched
        points = points.unsqueeze(0)  # (1, N, D)
    B, N, D = points.shape
    t = t.reshape(B, 1, D)
    q = q.reshape(B, 4)
    return rotate_vector_by_quaternion(points, q) + t


def yaw_from_R(R: torch.Tensor):
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def pitch_from_R(R: torch.Tensor):
    return torch.arcsin(-R[..., 2, 0])


def roll_from_R(R: torch.Tensor):
    return torch.atan2(R[..., 2, 1], R[..., 2, 2])


def rotation_matrix_to_euler_zyx(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert batch of rotation matrices to ZYX Euler angles (yaw, pitch, roll)
    Args:
        R: (B, 3, 3) batch of rotation matrices
    Returns:
        angles: (B, 3) tensor of Euler angles in radians
    """
    pitch = torch.asin(-R[:, 2, 0].clamp(-1 + eps, 1 - eps))
    # Safe cosine calculation
    cos_pitch = torch.cos(pitch)
    mask = torch.abs(cos_pitch) > eps
    yaw = torch.zeros_like(pitch)
    roll = torch.zeros_like(pitch)
    # Non-degenerate case
    yaw[mask] = torch.atan2(R[mask, 1, 0], R[mask, 0, 0])
    roll[mask] = torch.atan2(R[mask, 2, 1], R[mask, 2, 2])
    # Degenerate case (pitch near ±π/2)
    if torch.any(~mask):
        yaw[~mask] = 0.0
        roll[~mask] = torch.atan2(-R[~mask, 0, 1], R[~mask, 1, 1])
    return torch.stack([yaw, pitch, roll], dim=1)


def planar_rot_from_R3(R: torch.Tensor):
    ang = yaw_from_R(R)  # Extract yaw angle (rotation around Z axis)
    # Create the 2D rotation matrix for each batch element
    cos_ang = torch.cos(ang)
    sin_ang = torch.sin(ang)
    # Construct the planar rotation matrix (B, 2, 2)
    rot_matrix_2d = torch.stack(
        [torch.stack([cos_ang, -sin_ang], dim=-1), torch.stack([sin_ang, cos_ang], dim=-1)], dim=-2
    )  # Stack along new dimension to create (B, 2, 2)
    return rot_matrix_2d


def planar_rot_from_q(q: torch.Tensor):
    """
    Extracts the planar rotation matrix from a quaternion.
    q: Tensor of shape (N, 4)
    Returns: Tensor of shape (N, 2, 2)
    """
    # Extract the yaw angle from the quaternion
    yaw = quaternion_to_yaw(q)
    # Create the 2D rotation matrix for each batch element
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    # Construct the planar rotation matrix (B, 2, 2)
    rot_matrix_2d = torch.stack(
        [torch.stack([cos_yaw, -sin_yaw], dim=-1), torch.stack([sin_yaw, cos_yaw], dim=-1)], dim=-2
    )  # Stack along new dimension to create (B, 2, 2)
    return rot_matrix_2d


def quaternion_to_yaw(q):
    """
    Compute the yaw (psi) angle from a quaternion.

    Parameters:
    - q: Tensor of shape (..., 4), quaternion [w, x, y, z]

    Returns:
    - Yaw angle tensor in radians.
    """
    w, x, y, z = q.unbind(-1)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return yaw


def quaternion_to_pitch(q):
    """
    Compute the pitch (theta) angle from a quaternion.

    Parameters:
    - q: Tensor of shape (..., 4), quaternion [w, x, y, z]

    Returns:
    - Pitch angle tensor in radians.
    """
    w, x, y, z = q.unbind(-1)
    sinp = 2 * (w * y - z * x)
    sinp = sinp.clamp(-1, 1)
    pitch = torch.asin(sinp)
    return pitch


def quaternion_to_roll(q):
    """
    Compute the roll (phi) angle from a quaternion.

    Parameters:
    - q: Tensor of shape (..., 4), quaternion [w, x, y, z]

    Returns:
    - Roll angle tensor in radians.
    """
    w, x, y, z = q.unbind(-1)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    return roll


def euler_to_quaternion(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.

    Args:
        roll (torch.Tensor): Roll angle in radians.
        pitch (torch.Tensor): Pitch angle in radians.
        yaw (torch.Tensor): Yaw angle in radians.

    Returns:
        torch.Tensor: Quaternion represented as a tensor of shape (4,).
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    q = torch.stack([w, x, y, z], dim=-1)
    return normalized(q)


def quaternion_to_euler(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q (torch.Tensor): Quaternion represented as a tensor of shape (4,).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Roll, pitch, and yaw angles in radians.
    """
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(2 * (w * y - z * x))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def inverse_quaternion(q: torch.Tensor) -> torch.Tensor:
    q = q.view(-1, 4)
    s = torch.tensor([[1, -1, -1, -1]], device=q.device, dtype=q.dtype)
    return normalized(q * s)


def points_in_oriented_box(points: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """
    Check if points are inside an oriented box.

    Args:
        points (torch.Tensor): Tensor of shape (N, 2), where N is the number of 2D points.
        box (torch.Tensor): Tensor of shape (4, 2), representing the 4 corner points of the box in order.

    Returns:
        torch.Tensor: Boolean mask of shape (N,), where True indicates the point is inside the box.
    """
    assert points.shape[1] == 2, "Points must be 2D (N, 2)."
    assert box.shape == (4, 2), "Box must have 4 corner points with shape (4, 2)."
    edge_vectors = box.roll(-1, dims=0) - box  # Edges: [p1->p2, p2->p3, p3->p4, p4->p1]
    point_vectors = points.unsqueeze(1) - box.unsqueeze(0)  # Shape: (N, 4, 2)
    dot_prods = torch.sum(edge_vectors * point_vectors, dim=2)  # Shape: (N, 4)
    inside_mask = (dot_prods >= 0).all(dim=1) | (dot_prods <= 0).all(dim=1)
    return inside_mask


def bbox_limits_to_points(limits: torch.Tensor) -> torch.Tensor:
    """
    Convert 3D bounding box limits to

    Args:
        limits (torch.Tensor): Tensor of shape (6) representing the max and min coordinates of the bounding box.

    Returns:
        torch.Tensor: Tensor of shape (8, 3) representing the 8 corner points of the bounding box.
    """
    max_x, max_y, max_z = limits[:3]
    min_x, min_y, min_z = limits[3:]
    return torch.tensor(
        [
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ],
        device=limits.device,
        dtype=limits.dtype,
    )


def pointcloud_bounding_volume(pcd: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Determine the bounding volume of a point cloud.

    Args:
        pcd (torch.Tensor): Tensor of shape (N, 3), where N is the number of points.
        eps (float): A small margin value to enlarge the bounding volume.

    Returns:
        torch.Tensor - 8 points representing the corners of the bounding volume. (axis-aligned)
    """
    mins = pcd.min(dim=0).values
    maxs = pcd.max(dim=0).values
    mins -= eps
    maxs += eps
    # Create the 8 corners of the bounding volume
    return torch.tensor(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ],
        device=pcd.device,
        dtype=pcd.dtype,
    )


def extract_top_plane_from_box(box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given an axis-aligned bounding box in 3D, extract the top plane of the box.

    The returned normal direction is facing upwards (positive Z coordinate).

    Args :
        box (torch.Tensor): Axis-aligned bounding box in 3D. Shape (8, 3)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple containing the normal vector and a point on the plane.
    """
    max_x, max_y, max_z = box.max(dim=0).values
    min_x, min_y, min_z = box.min(dim=0).values
    top_midpoint = torch.tensor([(max_x + min_x) / 2, (max_y + min_y) / 2, max_z], device=box.device, dtype=box.dtype)
    normal = torch.tensor([0, 0, 1], device=box.device, dtype=box.dtype)
    return normal, top_midpoint


def points_within_circle(points: torch.Tensor, center: torch.Tensor, radius: float, eps: float = 0.0) -> torch.Tensor:
    """
    Check if points are within a circle in 2D.

    Args:
        points (torch.Tensor): Tensor of shape (N, 2), where N is the number of 2D points.
        center (torch.Tensor): Tensor of shape (2,), representing the center of the circle.
        radius (float): Radius of the circle.
        eps (float): Size of the margin around the radius.

    Returns:
        torch.Tensor: Boolean mask of shape (N,), where True indicates the point is within the circle.
    """
    assert points.shape[1] == 2, "Points must be 2D (N, 2)."
    assert center.shape == (2,), "Center must have shape (2,)."
    distances = torch.norm(points - center, dim=1)
    return (radius - eps <= distances) & (distances <= radius + eps)


def points_within_bbox(
    points: torch.Tensor,
    bbox: torch.Tensor,
) -> torch.Tensor:
    """
    Check if points are within an axis-aligned bounding box in D dimensions.

    Args:
        points (torch.Tensor): Tensor of shape (N, D), where N is the number of points and D is the number of dimensions.
        bbox (torch.Tensor): Tensor of shape (2*D,), representing the max and min coordinates of the bounding box.
    Returns:
        torch.Tensor: Boolean mask of shape (N,), where True indicates the point is within the bounding box.
    """
    assert points.dim() == 2, "Points must be 2D (N, D)."
    assert bbox.dim() == 1 and bbox.shape[0] % 2 == 0, "BBox must have shape (2*D,)."
    D = bbox.shape[0] // 2
    max_coords = bbox[:D]
    min_coords = bbox[D:]
    return torch.all((points >= min_coords) & (points <= max_coords), dim=1)
