from typing import Tuple

import torch

from .geometry import normalized

__all__ = [
    "make_x_y_grids",
    "compute_heightmap_gradients",
    "interpolate_normals",
    "interpolate_grid",
    "surface_normals_from_grads",
]


def make_x_y_grids(max_coord: float, grid_res: float, num_robots: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a grid of x and y coordinates.

    Args:
    - max_coord: Maximum coordinate value.
    - grid_res: Resolution of the grid in meters.
    - num_robots: Number of robots.

    Returns:
    - Tuple of x and y grids of shape (num_robots, D, D) where D = 2 * max_coord / grid_res.
    """
    dim = int(2 * max_coord / grid_res)
    xint = torch.linspace(-max_coord, max_coord, dim)  # haha
    yint = torch.linspace(-max_coord, max_coord, dim)
    x, y = torch.meshgrid(xint, yint, indexing="xy")
    x = x.unsqueeze(0).repeat(num_robots, 1, 1)
    y = y.unsqueeze(0).repeat(num_robots, 1, 1)
    return x, y


def compute_heightmap_gradients(z_grid: torch.Tensor, grid_res: float) -> torch.Tensor:
    """
    Computes the gradients of a heightmap along the x and y axes using torch.gradient.

    **Note**: The input grid is assumed to use torch's "xy" indexing convention, i.e., the first dimension is the y-axis and the second dimension is the x-axis.

    Parameters:
    - z_grid: Heightmap tensor of shape (B, H, W).
    - grid_res: Resolution of the grid in meters.
    """
    return torch.stack(torch.gradient(z_grid, spacing=grid_res, dim=(2, 1), edge_order=2), dim=1)


def surface_normals_from_grads(z_grid_grads: torch.Tensor, query: torch.Tensor, max_coord: float) -> torch.Tensor:
    """
    Computes the surface normals and tangents at the queried coordinates.

    Parameters:
    - z_grid_grads: torch.Tensor of gradients of the heightmap along the x and y axes (3D array), (B, 2, H, W).
    - query: Tensor of desired point coordinates for interpolation (3D array), (B, N, 2).

    Returns:
    - Surface normals at the queried coordinates.
    """
    norm_query = query / max_coord  # Normalize to [-1, 1]
    # Clamp the query coordinates to the grid's valid range
    norm_query = torch.clamp(norm_query, -1, 1)
    # Query coordinates of shape (B, N, 1, 2)
    B, N = query.shape[:2]
    grid_coords = norm_query.unsqueeze(2)
    # Interpolate the grid values into shape (B, 2, N, 1)
    grad_query = (
        torch.nn.functional.grid_sample(z_grid_grads, grid_coords, align_corners=True, mode="bilinear").squeeze(-1).transpose(1, 2)
    )  # (B, N, 2)
    # Compute the surface normals
    n = torch.dstack([-grad_query, torch.ones((B, N, 1), device=query.device)])  # n = [-dz/dx, -dz/dy, 1]
    n = normalized(n)
    return n


def interpolate_normals(normals: torch.Tensor, query: torch.Tensor, max_coord: float) -> torch.Tensor:
    """
    Interpolates the normals at the desired (query[0], query[1]]) coordinates.

    Parameters:
    - normals: Tensor of normals corresponding to the x and y coordinates (3D array), (B, 3, D, D). Top-left corner is (-max_coord, -max_coord). The indexing order follows the "xy" convention, meaning the first dimension is the y-axis and the second dimension is the x-axis.
    - query: Tensor of desired point coordinates for interpolation (3D array), (B, N, 2). Range is from -max_coord to max_coord.
    Returns:
    - Interpolated normals at the queried coordinates in shape (B, N, 3).
    """
    norm_query = query / max_coord  # Normalize to [-1, 1]
    # Clamp the query coordinates to the grid's valid range
    norm_query.clamp_(-1, 1)
    # Query coordinates of shape (B, N, 1, 2)
    grid_coords = norm_query.unsqueeze(2)
    # Interpolate the normals into shape (B, 3, N)
    interpolated_normals = torch.nn.functional.grid_sample(normals, grid_coords, align_corners=True, mode="bilinear").squeeze(3)
    return normalized(interpolated_normals.transpose(1, 2))


def interpolate_grid(grid: torch.Tensor, query: torch.Tensor, max_coord: float | torch.Tensor) -> torch.Tensor:
    """
    Interpolates the height at the desired (query[0], query[1]]) coordinates.

    Parameters:
    - grid: Tensor of grid values corresponding to the x and y coordinates (3D array), (B, D, D). Top-left corner is (-max_coord, -max_coord). The indexing order follows the "xy" convention, meaning the first dimension is the y-axis and the second dimension is the x-axis.
    - query: Tensor of desired point coordinates for interpolation (3D array), (B, N, 2). Range is from -max_coord to max_coord.
    Returns:
    - Interpolated grid values at the queried coordinates in shape (B, N, 1).
    """
    norm_query = query / max_coord  # Normalize to [-1, 1]
    # Clamp the query coordinates to the grid's valid range
    norm_query.clamp_(-1, 1)
    # Query coordinates of shape (B, N, 1, 2)
    grid_coords = norm_query.unsqueeze(2)
    # Grid of shape (B, 1, H, W)
    grid_w_c = grid.unsqueeze(1)
    # Interpolate the grid values into shape (B, 1, N, 1)
    z_query = torch.nn.functional.grid_sample(grid_w_c, grid_coords, align_corners=True, mode="bilinear")
    return z_query.squeeze(1)
