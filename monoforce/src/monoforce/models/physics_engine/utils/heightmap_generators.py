import torch
from dataclasses import dataclass
from typing import ClassVar
from abc import ABC, abstractmethod
from .environment import make_x_y_grids


@dataclass
class BaseHeightmapGenerator(ABC):
    """
    Base class for heightmap generators.

    Attributes:
    - add_random_noise: bool - whether to add random noise to the heightmap
    - noise_std: float - standard deviation of the Gaussian noise in meters
    - noise_mu: float - mean of the Gaussian noise in meters
    """

    add_random_noise: bool = False
    noise_std: float = 0.01  # Standard deviation of the Gaussian noise in meters
    noise_mu: float = 0.0  # Mean of the Gaussian noise in meters

    def __call__(
        self, grid_res: float, max_coord: float, num_robots: int, rng: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """
        Generates a heightmap.

        Args:
        - grid_res: Resolution of the grid in meters.
        - max_coord: Maximum coordinate in meters.
        - num_robots: Number of robots.
        - rng: Random number generator.

        Returns:
        - x: Tensor of x coordinates. Shape is (B, D, D).
        - y: Tensor of y coordinates. Shape is (B, D, D).
        - z: Heightmap tensor of shape (B, D, D).
        - mask: Suitability mask tensor of shape (B, D, D).
        """
        x, y = make_x_y_grids(max_coord, grid_res, num_robots)
        z, mask = self._generate_heightmap(x, y, max_coord, rng)
        if self.add_random_noise:
            z = self._add_noise_to_heightmap(z, rng)
        return x, y, z, mask

    @abstractmethod
    def _generate_heightmap(
        self, x: torch.Tensor, y: torch.Tensor, max_coord: float, rng: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a heightmap.

        Args:
        - x: Tensor of x coordinates. Shape is (D, D).
        - y: Tensor of y coordinates. Shape is (D, D).
        - max_coord: Maximum coordinate in meters.
        - rng: Random number generator.

        Returns:
        - Heightmap tensor of shape (D, D) and a suitability mask tensor of shape (D, D).
        """
        raise NotImplementedError

    def _add_noise_to_heightmap(self, z: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
        """
        Adds Gaussian noise to a heightmap.

        Args:
        - z: Heightmap tensor of shape (D, D).
        - rng: Random number generator.

        Returns:
        - Heightmap tensor of shape (D, D) with added noise.
        """
        noise = torch.normal(self.noise_mu, self.noise_std, size=z.shape, generator=rng, device=z.device)
        return z + noise


@dataclass
class MultiGaussianHeightmapGenerator(BaseHeightmapGenerator):
    """
    Generates a heightmap using multiple gaussians.
    """

    returns_suitability_mask: ClassVar[bool] = False
    min_gaussians: int = 30
    max_gaussians: int = 50
    min_height_fraction: float = 0.05
    max_height_fraction: float = 0.1
    min_std_fraction: float = 0.05
    max_std_fraction: float = 0.3
    min_sigma_ratio: float = 0.3

    def _generate_heightmap(
        self, x: torch.Tensor, y: torch.Tensor, max_coord: float, rng: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        z = torch.zeros_like(x)
        for i in range(B):
            # Generate random number of gaussians
            num_gaussians = int(torch.randint(self.min_gaussians, self.max_gaussians + 1, (1,), generator=rng).item())
            # Generate means from -max_coord to max_coord
            mus = torch.rand((2, num_gaussians), device=x.device) * 2 * max_coord - max_coord
            # Generate standard deviations from min_std_fraction * max_coord to max_std_fraction * max_coord
            sigmas = (
                torch.rand((num_gaussians, 1), device=x.device) * (self.max_std_fraction - self.min_std_fraction) * max_coord
                + self.min_std_fraction * max_coord
            )
            ratios = (
                torch.rand((num_gaussians,), device=x.device) * (1 - self.min_sigma_ratio) + self.min_sigma_ratio
            )  # ratio of the standard deviations of the x and y components in range [min_sigma_ratio, 1]
            higher_indices = torch.randint(0, 2, (num_gaussians,), device=x.device)  # whether the x or y component has the  higher standard deviation
            sigmas = sigmas.repeat(1, 2)
            sigmas[torch.arange(num_gaussians), higher_indices] *= ratios
            heights = torch.rand((num_gaussians,), device=x.device) * (self.max_height_fraction - self.min_height_fraction) + self.min_height_fraction
            z[i] = torch.sum(
                heights.view(-1, 1, 1)
                * torch.exp(
                    -(
                        (x[None, i] - mus[0].view(-1, 1, 1)) ** 2 / (2 * sigmas[..., 0].view(-1, 1, 1) ** 2)
                        + (y[None, i] - mus[1].view(-1, 1, 1)) ** 2 / (2 * sigmas[..., 1].view(-1, 1, 1) ** 2)
                    )
                ),
                dim=0,
            )
        return z, torch.ones_like(x, dtype=torch.bool, device=x.device)
