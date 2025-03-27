from dataclasses import dataclass
import torch
from .base_config import BaseConfig
from ..models.physics_engine.utils.environment import compute_heightmap_gradients


@dataclass
class WorldConfig(BaseConfig):
    """
    World configuration. Contains the physical constants of the world, coordinates of the world frame etc.

    **Note**:
        1) the convention for grids is torch's "xy" indexing of the meshgrid. This means that the first
            dimension of the grid corresponds to the y-coordinate and the second dimension corresponds to the x-coordinate. The coordinate increases with increasing index, i.e. the Y down and X right. Generate them with torch.meshgrid(y, x, indexing="xy").
        2) origin [0,0] is at the center of the grid.

    x_grid (torch.Tensor):  x-coordinates of the grid.
    y_grid (torch.Tensor): y-coordinates of the grid.
    z_grid (torch.Tensor): z-coordinates of the grid.
    z_grid_grad (torch.Tensor): gradients of the z-coordinates of the grid.
    normals (torch.Tensor): normals of the terrain. Shape (B, 3, grid_dim, grid_dim).
    grid_res (float): resolution of the grid in meters. Represents the metric distance between 2 centers of adjacent grid cells.
    max_coord (float): maximum coordinate of the grid.
    k_stiffness (float or torch.Tensor): stiffness of the terrain. Default is 20_000.
    k_friction (float or torch.Tensor): friction of the terrain. Default is 1.0.
    suitable_mask (torch.Tensor | None): mask of suitable terrain. Shape (grid_dim, grid_dim). 1 if suitable, 0 if not. Default is None.
    """

    x_grid: torch.Tensor
    y_grid: torch.Tensor
    z_grid: torch.Tensor
    grid_res: float
    max_coord: float
    k_stiffness: float | torch.Tensor = 20_000.0
    k_friction_lon: float | torch.Tensor = 0.5
    k_friction_lat: float | torch.Tensor = 0.2
    suitable_mask: torch.BoolTensor | None = None

    def __post_init__(self):
        self.z_grid_grad = compute_heightmap_gradients(self.z_grid, self.grid_res)  # (B, 2, D, D)
        ones = torch.ones_like(self.z_grid_grad[:, 0]).unsqueeze(1)
        self.normals = torch.cat((-self.z_grid_grad, ones), dim=1)
        self.normals /= torch.linalg.norm(self.normals, dim=1, keepdim=True)

    @property
    def grid_size(self) -> int:
        """
        Returns the size of the grid.

        Returns:
            int: size of the grid.
        """
        return self.z_grid.shape[-1]

    def ij_to_xyz(self, ij: torch.IntTensor | torch.LongTensor) -> torch.FloatTensor:
        """
        ij is assumed to have the shape (N, B, 2) where N is the number of points and B is the batch size.
        """
        N, B = ij.shape[:-1]
        linearized_idx = torch.cat(
            (
                torch.arange(
                    B,
                )
                .repeat(N)
                .unsqueeze(-1),
                ij.view(-1, 2),
            ),
            dim=-1,
        )
        x = self.x_grid[linearized_idx.unbind(-1)]
        y = self.y_grid[linearized_idx.unbind(-1)]
        z = self.z_grid[linearized_idx.unbind(-1)]
        return torch.stack((x, y, z), dim=-1).view(N, B, 3)

    def ij_to_suited_mask(self, ij: torch.IntTensor | torch.LongTensor) -> torch.BoolTensor:
        """
        ij is assumed to have the shape (N, B, 2) where N is the number of points and B is the batch size.
        """
        N, B = ij.shape[:-1]
        linearized_idx = torch.cat(
            (
                torch.arange(
                    B,
                )
                .repeat(N)
                .unsqueeze(-1),
                ij.view(-1, 2),
            ),
            dim=-1,
        )
        return self.suitable_mask[linearized_idx.unbind(-1)].view(N, B)
