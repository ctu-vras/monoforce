from dataclasses import dataclass
import torch
from ..utils.geometry import (
    normalized,
    points_within_circle,
    rodrigues_rotation_matrix,
)
from typing import Tuple


__all__ = ["get_track_pointwise_vels", "TrackWheels"]


@dataclass
class TrackWheels:
    """
    Represents the wheels of the track.

    Attributes:
        wheel_positions (torch.Tensor): Positions of the wheels in world coordinates. Shape: (N, 3)
        wheel_radii (torch.Tensor): Radii of the wheels. Shape: (N,)
        rotation_vector (torch.Tensor): Rotation vector of the flipper (rotation axis/de-facto signed surface normal of the wheels). Shape: (3)
    """

    wheel_positions: torch.Tensor
    wheel_radii: torch.Tensor
    rotation_vector: torch.Tensor

    @classmethod
    def from_dict(cls, data: dict) -> "TrackWheels":
        return cls(
            wheel_positions=torch.tensor(data["position"]),
            wheel_radii=torch.tensor(data["radius"]),
            rotation_vector=torch.tensor(data["rot_axis"]),
        )


def get_track_pointwise_vels(
    flipper_points: torch.Tensor,
    track_wheels: TrackWheels,
    forward_direction: torch.Tensor,
    wheel_assignment_margin: float,
    flat_min_dist: float,
) -> torch.Tensor:
    """
    From model of the track as multiple connected wheels, compute the velocity of the flipper points.

    Args:
        flipper_points (torch.Tensor): Flipper points in world coordinates. Shape: (N, 3)
        track_wheels (torch.Tensor): Wheels of the track, paramerized by their positions, radii, and rotation vector.
        forward_direction (torch.Tensor): Forward direction of the flipper. Used for sorting the wheels.
        wheel_assignment_margin (float): Margin for wheel assignment of points around the wheel.
    Returns:
        torch.Tensor: Velocity direction vectors of the flipper points.

    """
    N = track_wheels.wheel_positions.shape[0]
    # Find the rotation matrix of the flipper such that all of the points are in the XY plane
    wheel_rot_vec = track_wheels.rotation_vector
    rot_vec_norm = normalized(wheel_rot_vec)
    rot_axis = torch.cross(rot_vec_norm, torch.tensor([0.0, 0.0, 1.0]))
    theta = torch.acos(torch.dot(rot_vec_norm, torch.tensor([0.0, 0.0, 1.0])))
    R = rodrigues_rotation_matrix(rot_axis, theta).squeeze()
    rot_flipper_pts = torch.matmul(R, flipper_points.permute(1, 0)).permute(1, 0)
    # Sort the wheels based on their position in the forward direction
    wheel_dots = torch.sum(forward_direction.view(-1, 3) * track_wheels.wheel_positions, dim=1)
    sorting = torch.argsort(wheel_dots)
    rot_wheel_pos_sorted = torch.matmul(R, track_wheels.wheel_positions.T).T[sorting]
    wheel_radius_sorted = track_wheels.wheel_radii[sorting]
    vels = torch.zeros_like(flipper_points)
    used = torch.zeros(flipper_points.shape[0], dtype=torch.bool)
    for i in range(0, N, 2):
        diff_vecs, assigned = get_wheel_point_diff_vecs(
            rot_wheel_pos_sorted[i],
            wheel_radius_sorted[i].item(),
            rot_flipper_pts,
            used,
            wheel_assignment_margin,
        )
        # Assign the velocity direction vectors to the points as the cross product of the rotation vector and the diff vectors
        vels[assigned] = torch.cross(wheel_rot_vec.unsqueeze(0), (R.T @ diff_vecs[assigned].T).T)
        used[assigned] = True
        if i + 1 < N:  # If there is a next wheel
            diff_vecs, assigned = get_wheel_point_diff_vecs(
                rot_wheel_pos_sorted[i + 1],
                wheel_radius_sorted[i + 1].item(),
                rot_flipper_pts,
                used,
                wheel_assignment_margin,
            )
            vels[assigned] = torch.cross(wheel_rot_vec.unsqueeze(0), (R.T @ diff_vecs[assigned].T).T)
            used[assigned] = True
            # Now assign all points between the two wheels
            wheel_center_diff = rot_wheel_pos_sorted[i + 1] - rot_wheel_pos_sorted[i]
            wheel_center_diff[2] = 0
            remaining_pts_diffs_i = rot_flipper_pts - rot_wheel_pos_sorted[i]
            remaining_pts_diffs_i[:, 2] = 0
            remaining_pts_diffs_i_next = rot_flipper_pts - rot_wheel_pos_sorted[i + 1]
            remaining_pts_diffs_i_next[:, 2] = 0
            inbetween_mask = (torch.sum(remaining_pts_diffs_i.view(-1, 3) * wheel_center_diff, dim=1) > 0) & (
                torch.sum(remaining_pts_diffs_i_next.view(-1, 3) * wheel_center_diff, dim=1) < 0
            )
            v_u, v_v = tangent_vecs_between_wheels(
                rot_wheel_pos_sorted[i],
                wheel_radius_sorted[i].item(),
                rot_wheel_pos_sorted[i + 1],
                wheel_radius_sorted[i + 1].item(),
            )
            center_diff_normal = normalized(torch.tensor([wheel_center_diff[1], -wheel_center_diff[0], 0]))
            rot_flipper_pts_centered = rot_flipper_pts - (rot_wheel_pos_sorted[i] + wheel_center_diff / 2)
            rot_flipper_pts_centered[:, 2] = 0  # Ensure the points are in the XY plane
            centered_pts_dots = torch.sum(rot_flipper_pts_centered * center_diff_normal.view(-1, 3), dim=1)
            left_mask = centered_pts_dots > flat_min_dist
            right_mask = centered_pts_dots < -flat_min_dist
            remaining_mask = ~used
            vels[remaining_mask & inbetween_mask & left_mask] = R.T @ v_u
            vels[remaining_mask & inbetween_mask & right_mask] = R.T @ v_v
            used[remaining_mask & inbetween_mask] = True
    return normalized(vels)


def tangent_vecs_between_wheels(
    wheel_pos_a: torch.Tensor,
    wheel_radius_a: float,
    wheel_pos_b: torch.Tensor,
    wheel_radius_b: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the tangent vectors between two wheels.

    Args:
        wheel_pos_a (torch.Tensor): Position of the first wheel. Shape: (2,)
        wheel_radius_a (float): Radius of the first wheel.
        wheel_pos_b (torch.Tensor): Position of the second wheel. Shape: (2,)
        wheel_radius_b (float): Radius of the second wheel.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tangent vectors between the two wheels.
    """
    center_dist = torch.norm(wheel_pos_a - wheel_pos_b)
    center_diff = wheel_pos_b - wheel_pos_a
    theta = torch.atan2(center_diff[1], center_diff[0])
    phi = torch.asin(abs(wheel_radius_a - wheel_radius_b) / center_dist)

    ang_u = torch.tensor([theta + phi] * 2)
    ang_v = torch.tensor([theta - phi] * 2)
    if wheel_radius_a > wheel_radius_b:
        v_v = torch.tensor([-torch.cos(ang_u[0]), torch.sin(ang_u[0]), 0])
        v_u = torch.tensor([torch.cos(ang_v[0]), -torch.sin(ang_v[0]), 0])
    else:
        v_v = torch.tensor([-torch.cos(ang_v[0]), torch.sin(ang_v[0]), 0])
        v_u = torch.tensor([torch.cos(ang_u[0]), -torch.sin(ang_u[0]), 0])
    return v_u, v_v


def get_wheel_point_diff_vecs(
    wheel_position: torch.Tensor,
    wheel_radius: float,
    points: torch.Tensor,
    ignored_mask: torch.Tensor,
    wheel_assignment_margin: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Determine which points are within the wheel radius
    within_wheel = points_within_circle(points[:, :2], wheel_position[:2], wheel_radius, wheel_assignment_margin)
    assigned = torch.logical_and(within_wheel, ~ignored_mask)
    # Normal vectors from center of the wheel to the points
    diff_vecs = points - wheel_position
    diff_vecs[:, 2] = 0
    diff_vecs = normalized(diff_vecs)
    return diff_vecs, assigned
