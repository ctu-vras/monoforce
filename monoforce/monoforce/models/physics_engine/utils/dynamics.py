import torch

__all__ = ["inertia_tensor", "cog", "inertia_tensor_inv"]


def cog(pointwise_mass: torch.Tensor, points: torch.Tensor):
    """
    Compute the center of gravity of a rigid body represented by point masses.

    Parameters:

        mass (torch.Tensor): masses of the points in shape (N).
        points (torch.Tensor): A tensor of shape (B, N, 3) representing the points of the body.

    Returns:
        torch.Tensor: The center of gravity of the body.
    """
    return torch.sum(pointwise_mass[:, None] * points, dim=-2) / pointwise_mass.sum()


def inertia_tensor(pointwise_mass: torch.Tensor, points: torch.Tensor):
    """
    Compute the inertia tensor for a rigid body represented by point masses.

    Parameters:

        mass (torch.Tensor): masses of the points in shape (N).
        points (torch.Tensor): A tensor of shape (B, N, 3) representing the points of the body.
                            Each point contributes equally to the total mass.

    Returns:
        torch.Tensor: A 3x3 inertia tensor matrix.
    """
    points2mass = points * points * pointwise_mass[:, None]  # fuse this operation
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    x2m = points2mass[..., 0]
    y2m = points2mass[..., 1]
    z2m = points2mass[..., 2]
    Ixx = (y2m + z2m).sum(dim=-1)
    Iyy = (x2m + z2m).sum(dim=-1)
    Izz = (x2m + y2m).sum(dim=-1)
    Ixy = -(pointwise_mass[None, :] * x * y).sum(dim=-1)
    Ixz = -(pointwise_mass[None, :] * x * z).sum(dim=-1)
    Iyz = -(pointwise_mass[None, :] * y * z).sum(dim=-1)
    # Construct the inertia tensor matrix
    I = torch.stack([torch.stack([Ixx, Ixy, Ixz], dim=-1), torch.stack([Ixy, Iyy, Iyz], dim=-1), torch.stack([Ixz, Iyz, Izz], dim=-1)], dim=-2)  #
    return I


def inertia_tensor_inv(pointwise_mass: torch.Tensor, points: torch.Tensor):
    """
    Directly compute the inertia tensor inverse for a rigid body represented by point masses.

    Parameters:

        mass (torch.Tensor): masses of the points in shape (N).
        points (torch.Tensor): A tensor of shape (B, N, 3) representing the points of the body.
                            Each point contributes equally to the total mass.

    Returns:
        torch.Tensor: A 3x3 inverse inertia tensor matrix.
    """
    points2mass = points * points * pointwise_mass[:, None]  # fuse this operation
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    x2m = points2mass[..., 0]
    y2m = points2mass[..., 1]
    z2m = points2mass[..., 2]
    Ixx = (y2m + z2m).sum(dim=-1)
    Iyy = (x2m + z2m).sum(dim=-1)
    Izz = (x2m + y2m).sum(dim=-1)
    Ixy = -(pointwise_mass[None, :] * x * y).sum(dim=-1)

    Ixz = -(pointwise_mass[None, :] * x * z).sum(dim=-1)
    Iyz = -(pointwise_mass[None, :] * y * z).sum(dim=-1)
    # Construct the algebraic complement matrix
    A11 = Iyy * Izz - Iyz * Iyz
    A12 = Ixz * Iyz - Ixy * Izz
    A13 = Ixy * Iyz - Ixz * Iyy
    A21 = Iyz * Ixz - Ixy * Izz
    A22 = Ixx * Izz - Ixz * Ixz
    A23 = Ixy * Ixz - Ixx * Iyz
    A31 = Ixy * Iyz - Iyy * Ixz
    A32 = Ixy * Ixz - Iyz * Ixx
    A33 = Ixx * Iyy - Ixy * Ixy
    detA = Ixx * A11 + Ixy * A12 + Ixz * A13
    I_inv = (
        torch.stack([torch.stack([A11, A12, A13], dim=-1), torch.stack([A21, A22, A23], dim=-1), torch.stack([A31, A32, A33], dim=-1)], dim=-2)
        / detA[:, None, None]
    )
    return I_inv
