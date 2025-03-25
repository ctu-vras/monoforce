import torch
from .geometry import skew_symmetric, normalized, quaternion_multiply

__all__ = [
    "integrate_rotation",
    "condition_rotation_matrices",
    "integrate_quaternion",
]


def integrate_rotation(
    R: torch.Tensor,
    omega: torch.Tensor,
    dt: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Integrates the rotation matrix for the next time step using Rodrigues' formula.
    Parameters:
    - R: Tensor of rotation matrices. Shape [B, 3, 3]
    - omega: Tensor of angular velocities. Shape [B, 3]
    - dt: Time step. Float value.
    - eps: Small value to avoid division by zero. Float value.

    Returns:
    - Updated rotation matrices. Shape [B, 3, 3]
    """
    delta_omega = omega * dt  # Shape: [B, 3]
    theta = torch.norm(delta_omega, dim=1, keepdim=True)  # Rotation angle
    omega_skew = skew_symmetric(delta_omega)
    Id = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)  # Shape: [1, 3, 3]
    theta_expand = torch.clamp(theta.unsqueeze(2), eps)
    sin_term = torch.sin(theta_expand) / theta_expand
    cos_term = (1 - torch.cos(theta_expand)) / (theta_expand**2)
    omega_skew_squared = torch.bmm(omega_skew, omega_skew)
    delta_R = Id + sin_term * omega_skew + cos_term * omega_skew_squared
    return torch.bmm(delta_R, R)


def condition_rotation_matrices(R: torch.Tensor) -> torch.Tensor:
    """
    Condition the rotation matrices to prevent numerical instability.
    This is done by performing SVD on the rotation matrices and then reconstructing them.

    Parameters:
        - R: Rotation matrices. Shape [B, 3, 3].

    Returns:
        - torch.Tensor: Conditioned rotation matrices. Shape [B, 3, 3].
    """
    U, _, V = torch.svd(R)
    R = torch.bmm(U, V.transpose(-2, -1))
    return R


def integrate_quaternion(q: torch.Tensor, omega: torch.Tensor, dt: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Integrate quaternion using the quaternion derivative.
    Parameters:
        - q: Quaternion. Shape [B, 4].
        - omega: Angular velocity. Shape [B, 3].
        - dt: Time step. Float value.
    Returns:
        - Updated quaternion. Shape [B, 4].
    """

    half_dt_omega = 0.5 * dt * omega  # shape: (B, 3)

    # Theta is the magnitude of this half increment.
    theta = torch.norm(half_dt_omega, dim=1, keepdim=True)  # shape: (B, 1)

    # Compute sin(theta)/theta using a safe expression to avoid division by zero.
    # Where theta is very small, we use the fact that sin(theta)/theta ~ 1.
    sin_theta_over_theta = torch.where(theta > eps, torch.sin(theta) / theta, 1.0)  # shape: (B, 1)

    # The vector part of the rotation delta quaternion.
    delta_q_vector = sin_theta_over_theta * half_dt_omega  # shape: (B, 3)

    # The scalar part is simply cos(theta)
    delta_q_scalar = torch.cos(theta)  # shape: (B, 1)

    # Construct the delta quaternion. (Here we use the convention [w, x, y, z])
    delta_q = torch.cat((delta_q_scalar, delta_q_vector), dim=1)

    # The new quaternion is the product of delta_q and the current orientation.
    # (Using left-multiplication: q_new = delta_q ⊗ q)
    q_new = quaternion_multiply(delta_q, q)

    # Normalize the quaternion to protect against numerical drift.
    q_new = normalized(q_new)

    return q_new
