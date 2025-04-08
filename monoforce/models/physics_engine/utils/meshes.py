from typing import Literal, Tuple
import torch
import trimesh
import pyvista as pv
import pyacvd
import numpy as np
from ..utils.dynamics import inertia_tensor, cog

__all__ = [
    "cluster_points",
    "extract_surface_from_mesh",
    "voxelize_mesh",
    "extract_submesh_by_mask",
    "sample_points_from_convex_hull",
]


def cluster_points(points: torch.Tensor, n_points: int, **clus_opts) -> torch.Tensor:
    """
    Clusters a point cloud to n_points using the pyacvd library.

    Args:
        points (torch.Tensor): point cloud.
        n_points (int): number of points to cluster to.

    Returns:
        torch.Tensor: clustered points.
    """
    surf = pv.PolyData(points.numpy()).delaunay_3d(progress_bar=True)
    surf = surf.extract_geometry().triangulate()
    clus = pyacvd.Clustering(surf)
    clus.cluster(n_points, **clus_opts)
    return torch.tensor(clus.cluster_centroid)


def extract_surface_from_mesh(mesh: pv.PolyData, n_points: int = 100, **clus_opts) -> torch.Tensor:
    """
    Extracts the surface of a mesh and clusters it to n_points.

    First, the delauany triangulation is computed and the surface is extracted.
    Then, the surface is clustered using the pyacvd library.

    Args:
        mesh (pv.PolyData): mesh object.
        n_points (int, optional): number of points extracted. Defaults to 100.

    Returns:
        torch.Tensor: extracted points.
    """
    delaunay = mesh.delaunay_3d()
    surf = delaunay.extract_surface()
    clus: pyacvd.Clustering = pyacvd.Clustering(surf)
    clus.cluster(n_points, **clus_opts)
    return torch.tensor(clus.cluster_centroid)


def voxelize_mesh(mesh: pv.PolyData, voxel_size: float) -> torch.Tensor:
    """
    Voxelizes a mesh and returns the voxelized points.

    Args:
        mesh (pv.PolyData): mesh object.
        voxel_size (float): size of the voxel.

    Returns:
        torch.Tensor: voxelized points

    """
    mesh = pv.voxelize(mesh, voxel_size)
    return torch.tensor(mesh.points)


def extract_submesh_by_mask(mesh: pv.PolyData, mask: torch.Tensor) -> pv.PolyData:
    """
    Extracts a submesh from a mesh using a boolean mask.

    Args:
        mesh (pv.PolyData): mesh object.
        mask (torch.Tensor): boolean mask of the points to extract.

    Returns:
        pv.PolyData: extracted submesh.
    """
    indices = mask.nonzero().flatten().numpy()
    return mesh.extract_points(indices, adjacent_cells=False, include_cells=True)


def inertia_cog_from_voxelized_mesh(mesh: pv.PolyData, mass: float, voxel_size: float, fill: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the inertia tensor and center of gravity from a voxelized mesh.

    Args:
        mesh (pv.PolyData): mesh object.
        mass (float): mass of the object.
        voxel_size (float): size of the voxel.
        fill (bool, optional): whether to fill the voxels. Defaults to True.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: inertia tensor and center of gravity.
    """
    vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    vox = trimesh.voxel.creation.voxelize(mesh, pitch=voxel_size)
    if fill:
        filled_encoding = trimesh.voxel.morphology.binary_closing(vox.encoding, structure=np.ones((3, 3, 3)))
        vox = trimesh.voxel.VoxelGrid(filled_encoding, transform=vox.transform)
    points = torch.tensor(vox.points)
    pointwise_mass = mass / points.shape[0] * torch.ones(points.shape[0])
    cog_coords = cog(pointwise_mass, points)
    points -= cog_coords
    inertia_tensor_matrix = inertia_tensor(pointwise_mass, points.unsqueeze(0)).squeeze()
    return inertia_tensor_matrix, cog_coords


def sample_points_from_convex_hull(mesh: pv.PolyData, n_points: int, method: Literal["regular", "even"] = "even") -> torch.Tensor:
    """
    Samples points from the surface of the convex hull of a mesh.

    Args:
        mesh (pv.PolyData): mesh object.
        n_points (int): number of points to sample.

    Returns:
        torch.Tensor: sampled points.
    """
    mesh = mesh.extract_surface().triangulate()
    vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    ch = mesh.convex_hull
    if method == "regular":
        points = trimesh.sample.sample_surface(ch, n_points)[0]
    elif method == "even":
        points = trimesh.sample.sample_surface_even(ch, n_points)[0]
    else:
        raise ValueError(f"Method {method} not supported.")
    return torch.tensor(points)
