from typing import Union, List, Tuple
from pathlib import Path
import hashlib
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pyvista as pv
import torch
import yaml
from .base_config import BaseConfig
from ..models.physics_engine.utils.flipper_modeling import (
    TrackWheels,
    get_track_pointwise_vels,
)
from ..models.physics_engine.utils.geometry import bbox_limits_to_points, points_within_bbox
from ..models.physics_engine.utils.meshes import (
    extract_submesh_by_mask,
    extract_surface_from_mesh,
    inertia_cog_from_voxelized_mesh,
    sample_points_from_convex_hull,
)

# np.random.seed(0)

ROOT = Path(__file__).parent.parent.parent / "config"
MESHDIR = ROOT / "meshes"
YAMLDIR = ROOT / "robots"
POINTCACHE = ROOT / ".robot_cache"


def list_of_dicts_to_dict_of_lists(list_of_dicts: List[dict]) -> dict:
    """
    Converts a list of dictionaries to a dictionary of lists.

    Args:
        list_of_dicts (list[dict]): List of dictionaries.

    Returns:
        dict: Dictionary of lists.
    """
    return {key: [d[key] for d in list_of_dicts] for key in list_of_dicts[0].keys()}


@dataclass
class RobotModelConfig(BaseConfig):
    """
    Configuration of the robot model. Contains the physical constants of the robot, its mass and geometry.
    """

    kind: Literal["tradr", "marv", "husky"] = "marv"
    mesh_voxel_size: float = 0.01
    points_per_driving_part: int = 128
    points_per_body: int = 256
    wheel_assignment_margin: float = 0.02
    linear_track_assignment_margin: float = 0.05

    def __post_init__(self):
        self.load_robot_params_from_yaml()
        self.create_robot_geometry()
        self.disable_grads()
        # print(self)

    def disable_grads(self):
        """
        Disables gradients for all tensors in the dataclass.
        """
        for _, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                val.requires_grad = False

    def load_robot_params_from_yaml(self) -> None:
        """
        Loads the robot parameters from a yaml file.

        Parameters:
            robot_type (str): Name of the robot.

        Returns:
            None
        """
        with open(YAMLDIR / f"{self.kind}.yaml", "r") as file:
            robot_params = yaml.safe_load(file)
            canonical = yaml.dump(robot_params, sort_keys=True)  # ensure consistent order
        self.yaml_hash = hashlib.sha256(canonical.encode()).hexdigest()
        self.v_max = robot_params["v_max"]
        self.driving_direction = torch.tensor(robot_params["forward_direction"])
        self.body_mass = robot_params["body"]["mass"]
        self.body_bbox = torch.tensor(robot_params["body"]["bbox"]) if "bbox" in robot_params["body"] else None
        self.num_driving_parts = len(robot_params["driving_parts"])
        driving_parts = list_of_dicts_to_dict_of_lists(robot_params["driving_parts"])
        self.driving_part_bboxes = torch.tensor(driving_parts["bbox"])
        self.driving_part_masses = torch.tensor(driving_parts["mass"])
        self.track_wheels = [TrackWheels.from_dict(wheel) for wheel in driving_parts["wheels"]]
        self.joint_positions = torch.tensor(driving_parts["joint_position"])
        self.joint_limits = torch.tensor(driving_parts["joint_limits"]).T  # transpose from shape (num_driving_parts, 2) to (2, num_driving_parts)
        self.joint_max_pivot_vels = torch.tensor(driving_parts["max_pivot_vel"])
        self.total_mass = self.body_mass + self.driving_part_masses.sum().item()
        self.mesh_file = robot_params["mesh"]
        self.driving_part_movable_mask = torch.tensor(driving_parts["is_movable"]).float()

    def __repr__(self) -> str:
        s = f"RobotModelConfig for {self.kind}"
        s += f"\nBody mass: {self.body_mass}"
        s += f"\nTotal mass: {self.total_mass}"
        s += f"\nBody bbox: {self.body_bbox}"
        s += f"\nNumber of driving parts: {self.num_driving_parts}"
        s += f"\nDriving part masses: {self.driving_part_masses}"
        s += f"\nDriving part bboxes: {self.driving_part_bboxes}"
        s += f"\nJoint positions: {self.joint_positions}"
        s += f"\nJoint limits: {self.joint_limits}"
        s += f"\nJoint max pivot vels: {self.joint_max_pivot_vels}"
        s += f"\nTrack wheels: {self.track_wheels}"
        s += f"\nMax velocity: {self.v_max}"
        s += f"\nDriving direction: {self.driving_direction}"
        s += f"\nBody voxel size: {self.mesh_voxel_size}"
        s += f"\nPoints per driving part: {self.points_per_driving_part}"
        s += f"\nWheel assignment margin: {self.wheel_assignment_margin}"
        s += f"\nLinear track assignment margin: {self.linear_track_assignment_margin}"
        s += f"\nTotal number of points: {self.points_per_body + self.points_per_driving_part * self.num_driving_parts}"
        # self._print_tensor_info()
        return s

    @property
    def _descr_str(self) -> str:
        return f"{self.kind}_{self.mesh_voxel_size:.3f}_dp{self.points_per_driving_part}b_{self.points_per_body}_whl{self.wheel_assignment_margin}_trck{self.linear_track_assignment_margin}_{self.yaml_hash}"

    def _print_tensor_info(self):
        for name, tensor in self.__dict__.items():
            if isinstance(tensor, torch.Tensor):
                print(f"{name}: {tensor.shape}")

    def vw_to_vels(self, v: Union[float, torch.Tensor], w: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Linear/angular velocity to track wheel velocities.
        """
        if isinstance(v, float):
            v = torch.tensor(v)
        if isinstance(w, float):
            w = torch.tensor(w)
        v = v.view(-1, 1)
        w = w.view(-1, 1)
        driving_direction_2d = self.driving_direction[:2]
        driving_direction_2d = driving_direction_2d / torch.linalg.norm(driving_direction_2d)
        joint_positions_2d = self.joint_positions[:, :2]
        joint_dist_non_driving = (
            joint_positions_2d - torch.sum(joint_positions_2d * driving_direction_2d, dim=-1, keepdim=True) * driving_direction_2d
        )
        joint_dist_non_driving *= self.driving_direction[None, [1, 0]] * torch.tensor([1, -1], device=self.driving_direction.device)  # normal of the driving direction
        joint_dist_non_driving = joint_dist_non_driving.sum(dim=-1)
        vels = v.repeat(1, self.num_driving_parts)  # shape (batch_size, num_driving_parts)
        vels += w * joint_dist_non_driving  # shape (batch_size, num_driving_parts)
        return vels.clamp(-self.v_max, self.v_max)

    def load_from_cache(self) -> bool:
        """
        Loads the robot parameters from a cache file.

        Returns:
            bool: True if the cache file exists, False otherwise.
        """
        confpath = POINTCACHE / self._descr_str
        if confpath.exists():
            print(f"Loading robot model from cache: {confpath}")
            confdict = torch.load(confpath)
            if confdict["yaml_hash"] != self.yaml_hash:
                print("Hash mismatch, re-creating robot model")
                return False
            for key, val in confdict.items():
                setattr(self, key, val)
            return True
        return False

    def save_to_cache(self) -> None:
        """
        Saves the robot parameters to a cache file.
        """
        confpath = POINTCACHE / self._descr_str
        if not confpath.parent.exists():
            confpath.parent.mkdir(parents=True)
        print(f"Saving robot model to cache: {confpath}")
        confdict = {
            "yaml_hash": self.yaml_hash,
            "driving_part_points": self.driving_part_points,
            "driving_part_inertias": self.driving_part_inertias,
            "driving_part_cogs": self.driving_part_cogs,
            "body_points": self.body_points,
            "body_inertia": self.body_inertia,
            "body_cog": self.body_cog,
            "radius": self.radius,
            "thrust_directions": self.thrust_directions,
            "joint_local_driving_part_pts": self.joint_local_driving_part_pts,
            "joint_local_driving_part_cogs": self.joint_local_driving_part_cogs,
        }
        torch.save(confdict, confpath)

    def _construct_driving_parts(
        self,
        robot_mesh: pv.PolyData,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Constructs the driving parts of the robot
        Args:
            robot_points (torch.Tensor): Points of the robot.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Points, inertias, center of gravity of the driving parts and masks.
        """
        robot_points = torch.tensor(robot_mesh.points)
        driving_part_points = []
        driving_part_inertias = []
        driving_part_cogs = []
        driving_part_masks = []
        # Create surface meshes for the driving parts
        for i in range(self.num_driving_parts):
            mask = points_within_bbox(robot_points, bbox=self.driving_part_bboxes[i])
            driving_mesh = extract_submesh_by_mask(robot_mesh, mask).extract_surface()
            inertia, cog_coords = inertia_cog_from_voxelized_mesh(driving_mesh, self.driving_part_masses[i].item(), self.mesh_voxel_size, fill=True)
            driving_points = extract_surface_from_mesh(driving_mesh, n_points=self.points_per_driving_part)
            driving_part_points.append(driving_points)
            driving_part_inertias.append(inertia)
            driving_part_cogs.append(cog_coords)
            driving_part_masks.append(mask)
        return (
            torch.stack(driving_part_points).float(),
            torch.stack(driving_part_inertias).float(),
            torch.stack(driving_part_cogs).float(),
            torch.stack(driving_part_masks).float(),
        )

    def _construct_body(self, mesh: pv.PolyData, driving_part_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        robot_points = torch.tensor(mesh.points)
        body_mask = points_within_bbox(robot_points, self.body_bbox) if self.body_bbox is not None else torch.sum(driving_part_masks, dim=0) == 0
        robot_body = extract_submesh_by_mask(mesh, body_mask).extract_surface()
        robot_body_points = sample_points_from_convex_hull(robot_body, self.points_per_body, method="even")
        inertia, cog_coords = inertia_cog_from_voxelized_mesh(robot_body, self.body_mass, self.mesh_voxel_size, fill=True)
        return robot_body_points.float(), inertia.float(), cog_coords.float()

    def create_robot_geometry(self) -> None:
        if self.load_from_cache():
            return
        # Load the mesh and voxelized mesh
        mesh = pv.read(MESHDIR / self.mesh_file)
        # Construct the driving parts and body, compute their inertias and center of gravity
        (
            self.driving_part_points,  # shape (num_driving_parts, points_per_driving_part, 3)
            self.driving_part_inertias,  # shape (num_driving_parts, 3, 3)
            self.driving_part_cogs,  # shape (num_driving_parts, 3)
            driving_part_masks,  # shape (num_driving_parts, num_points)
        ) = self._construct_driving_parts(mesh)
        self.body_points, self.body_inertia, self.body_cog = self._construct_body(mesh, driving_part_masks)
        self.joint_local_driving_part_pts = self.driving_part_points - self.joint_positions[:, None, :]
        self.joint_local_driving_part_cogs = self.driving_part_cogs - self.joint_positions
        self.radius = torch.linalg.norm(torch.cat([self.body_points, *self.driving_part_points], dim=0)[..., :2], dim=-1).max().item()
        self.thrust_directions = self._calculate_thrust_directions()
        # Save to cache
        self.save_to_cache()

    def _calculate_thrust_directions(self) -> torch.Tensor:
        """
        Calculate the thrust directions for the robot points.

        Returns:
            torch.Tensor: Thrust directions for the robot points.
        """
        thrust_directions = torch.zeros_like(self.driving_part_points)
        for i in range(self.num_driving_parts):
            thrust_directions[i] = -get_track_pointwise_vels(
                self.driving_part_points[i],
                self.track_wheels[i],
                self.driving_direction,
                self.wheel_assignment_margin,
                self.linear_track_assignment_margin,
            )
        return thrust_directions

    def visualize_robot(
        self,
        grid_size: float = 1.0,
        grid_spacing: float = 0.1,
        return_plotter: bool = False,
    ) -> None:
        """
        Visualizes the robot in 3D using PyVista.

        Parameters:
            grid_size (float): Size of the grid.
            grid_spacing (float): Spacing of the grid.
            return_plotter (bool): Return the plotter object.
            jupyter_backend (str): Jupyter backend.
        """
        body_points = self.body_points.cpu().numpy()
        driving_part_points = self.driving_part_points.cpu().numpy()
        thrust_directions = self.thrust_directions.cpu().numpy()
        plotter = pv.Plotter()
        for i in range(self.num_driving_parts):
            driving_part_pcd = pv.PolyData(driving_part_points[i])
            driving_part_pcd["vectors"] = thrust_directions[i] * 0.1
            driving_part_pcd.set_active_vectors("vectors")
            plotter.add_mesh(driving_part_pcd, color="red", point_size=5, render_points_as_spheres=True)
            plotter.add_mesh(driving_part_pcd.arrows, color="black", opacity=0.15)
        body_pcd = pv.PolyData(body_points)
        plotter.add_mesh(body_pcd, color="blue", point_size=5, render_points_as_spheres=True)
        # joint positions
        for joint in self.joint_positions.cpu():
            sphere = pv.Sphere(center=joint.numpy(), radius=0.02)
            plotter.add_mesh(sphere, color="green")
        # bounding volumes - body
        if self.body_bbox is not None:
            body_bounding_volume_mesh = pv.PolyData(bbox_limits_to_points(self.body_bbox.cpu()).numpy()).delaunay_3d()
            plotter.add_mesh(body_bounding_volume_mesh, color="blue", line_width=2, opacity=0.1)
        # bounding volumes - driving parts
        for box in self.driving_part_bboxes.cpu():
            driving_part_bounding_volume_mesh = pv.PolyData(bbox_limits_to_points(box).numpy()).delaunay_3d()
            plotter.add_mesh(driving_part_bounding_volume_mesh, color="red", line_width=2, opacity=0.1)
        # origin of robot's local frame
        sphere = pv.Sphere(center=np.zeros(3), radius=0.03)
        plotter.add_mesh(sphere, color="yellow")
        # grid
        for i in np.arange(-grid_size, grid_size + grid_spacing, grid_spacing):
            line_x = pv.Line(pointa=[i, -grid_size, 0], pointb=[i, grid_size, 0])
            line_y = pv.Line(pointa=[-grid_size, i, 0], pointb=[grid_size, i, 0])
            plotter.add_mesh(line_x, color="black", line_width=0.5)
            plotter.add_mesh(line_y, color="black", line_width=0.5)
        # driving direction arrow
        driving_direction_arrow = pv.Arrow(
            direction=self.driving_direction.cpu().numpy(),
            tip_length=0.1,
            tip_radius=0.01,
            shaft_radius=0.005,
        )
        plotter.add_mesh(driving_direction_arrow, color="green")
        # show
        # print(self)
        plotter.show_axes()
        if not return_plotter:
            plotter.show()
        else:
            return plotter


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_type", type=str, default="marv", help="Type of the robot")
    parser.add_argument("--mesh_voxel_size", type=float, default=0.01, help="Voxel size")
    parser.add_argument("--points_per_driving_part", type=int, default=192, help="Number of points per driving part")
    args = parser.parse_args()
    robot_model = RobotModelConfig(
        kind=args.robot_type,
        mesh_voxel_size=args.mesh_voxel_size,
        points_per_driving_part=args.points_per_driving_part,
    )
    robot_model.visualize_robot()
