from __future__ import annotations
from typing import Tuple
import torch
from monoforce.configs.engine_config import PhysicsEngineConfig
from monoforce.configs import RobotModelConfig
from monoforce.configs import WorldConfig
from ..engine.engine_state import AuxEngineInfo, PhysicsState, PhysicsStateDer
from ..utils.environment import interpolate_grid, surface_normals_from_grads
from ..utils.geometry import q_to_R, rot_Y, normalized, rotate_vector_by_quaternion
from ..utils.numerical import integrate_quaternion


class DPhysicsEngine(torch.nn.Module):
    def __init__(self, config: PhysicsEngineConfig, robot_model: RobotModelConfig, device: torch.device | str):
        super().__init__()
        self.config = config
        self.device = device
        self.robot_model = robot_model
        self._F_g = torch.tensor(
            [0.0, 0.0, -self.robot_model.total_mass * self.config.gravity],
            device=self.device,
            requires_grad=False,
        )
        self._I_3x3 = torch.eye(3, device=self.device, requires_grad=False).view(1, 1, 1, 3, 3)

    def forward(self, state: PhysicsState, controls: torch.Tensor, world_config: WorldConfig) -> tuple[PhysicsState, PhysicsStateDer, AuxEngineInfo]:
        """
        Forward pass of the physics engine.
        """
        state_der, aux_info = self.forward_kinematics(state, controls, world_config)
        return self.update_state(state, state_der), state_der, aux_info

    def forward_kinematics(self, state: PhysicsState, controls: torch.Tensor, world_config: WorldConfig) -> Tuple[PhysicsStateDer, AuxEngineInfo]:
        robot_points, global_thrust_vectors, global_cogs, inertia = self.assemble_and_transform_robot(state, controls)

        # find the contact points
        in_contact, dh_points, n = self.find_contact_points(robot_points, world_config)

        # Compute current point velocities based on CoG motion and rotation
        cog_corrected_points = robot_points - global_cogs  # shape (B, n_pts, 3)
        xd_points = state.xd.unsqueeze(1) + torch.cross(state.omega.unsqueeze(1), cog_corrected_points, dim=-1)

        F_spring = self.calculate_spring_force(
            dh_points,
            xd_points,
            in_contact,
            n,
            world_config,
        )
        # friction forces
        k_friction_lon = world_config.k_friction_lon
        k_friction_lat = world_config.k_friction_lat
        F_friction = self.calculate_friction(
            state.q,
            F_spring,
            xd_points,
            global_thrust_vectors,
            n,
            k_friction_lon,
            k_friction_lat,
        )

        act_force = F_spring + F_friction  # total force acting on the robot's points
        torque, omega_d = self.calculate_torque_omega_d(act_force, cog_corrected_points, inertia)

        # motion of the cog
        F_cog = self._F_g + act_force.sum(dim=1)  # F = F_spring + F_friction + F_grav
        xdd = F_cog / self.robot_model.total_mass  # a = F / m, very funny xdd

        # joint rotational velocities, shape (B, n_joints)
        thetas_d = self.compute_joint_angular_velocities(controls)

        # next state derivative
        next_state_der = PhysicsStateDer(xd=state.xd, xdd=xdd, omega_d=omega_d, thetas_d=thetas_d, batch_size=[self.config.num_robots])

        # auxiliary information (e.g. for visualization)
        aux_info = AuxEngineInfo(
            F_spring=F_spring,
            F_friction=F_friction,
            in_contact=in_contact,
            torque=torque,
            global_robot_points=robot_points,
            global_thrust_vectors=global_thrust_vectors,
            batch_size=[self.config.num_robots],
        )
        return next_state_der, aux_info

    def calculate_spring_force(
        self,
        dh_points: torch.Tensor,
        xd_points: torch.Tensor,
        in_contact: torch.Tensor,
        n: torch.Tensor,
        world_config: WorldConfig,
    ) -> torch.Tensor:
        """
        Calculate the spring force acting on the robot points.
        """
        num_contacts = in_contact.sum(dim=1, keepdim=True).clamp_min(1)  # shape (B, 1, 1)
        k_damping = self.config.damping_alpha * 2 * (self.robot_model.total_mass * world_config.k_stiffness / num_contacts) ** 0.5
        # F_s = -k * dh - b * v_n, multiply by -n to get the force vector
        xd_points_n = (xd_points * n).sum(dim=-1, keepdim=True)  # normal component of the velocity
        F_spring = -torch.mul((world_config.k_stiffness * dh_points + k_damping * xd_points_n), n)
        return F_spring * in_contact / num_contacts  # shape (B, n_pts, 3), the spring force acting on the robot points

    def compute_joint_angular_velocities(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Compute the joint angle velocities based on the control inputs and the current joint angles.

        Args:
            controls: The control inputs.
            thetas: The current joint angles.

        Returns:
            thetas_d: The joint angle velocities.
        """
        thetas_d = controls[:, self.robot_model.num_driving_parts :]
        thetas_d = thetas_d.clamp(-self.robot_model.joint_max_pivot_vels, self.robot_model.joint_max_pivot_vels)
        return thetas_d

    def calculate_torque_omega_d(
        self, act_force: torch.Tensor, cog_corrected_points: torch.Tensor, inertia: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the angular acceleration of the robot.

        rigid body rotation: M = sum(r_i x F_i)

        Args:
            act_force: The total force acting on the robot's points.
            cog_corrected_points: The CoG corrected robot points in global coordinates.
            inertia: The inertia tensor in the global frame.

        Returns:
            omega_d: The angular acceleration of the robot.
        """
        torque = torch.sum(torch.cross(cog_corrected_points, act_force, dim=-1), dim=1)
        torque = torch.clamp(torque, -self.config.torque_limit, self.config.torque_limit)
        omega_d = torch.linalg.solve_ex(inertia, torque)[0]
        return torque, omega_d

    def calculate_friction(
        self,
        q: torch.Tensor,
        F_normal: torch.Tensor,
        xd_points: torch.Tensor,
        thrust_vectors: torch.Tensor,
        n: torch.Tensor,
        k_friction_lon: float | torch.Tensor,
        k_friction_lat: float | torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the friction force acting on the robot points.
        """

        # friction forces: shttps://en.wikipedia.org/wiki/Friction
        N = torch.norm(
            F_normal, dim=2, keepdim=True
        )  # normal force magnitude at the contact points, guaranteed to be zero if not in contact because of the spring force being zero
        global_driving_dir = rotate_vector_by_quaternion(self.robot_model.driving_direction.expand(self.config.num_robots, 1, 3), q)
        forward_dir = normalized(global_driving_dir - (global_driving_dir * n).sum(dim=-1, keepdims=True) * n)  # forward direction
        lateral_dir = normalized(torch.cross(forward_dir, n, dim=-1))  # lateral direction
        dv = thrust_vectors - xd_points  # velocity difference between the commanded and the actual velocity of the robot points
        dv_n = (dv * n).sum(dim=-1, keepdims=True)  # normal component of the relative velocity computed as dv_n = dv . n
        dv_tau = dv - dv_n * n  # tangential component of the relative velocity
        dv_tau = torch.tanh(dv_tau)  # saturate the tangential velocity difference
        dv_lon = (dv_tau * forward_dir).sum(dim=-1, keepdim=True) * forward_dir
        dv_lat = (dv_tau * lateral_dir).sum(dim=-1, keepdim=True) * lateral_dir
        F_friction_lon = k_friction_lon * N * dv_lon  # longitudinal friction force
        F_friction_lat = k_friction_lat * N * dv_lat  # lateral friction force
        F_friction = F_friction_lat + F_friction_lon
        return F_friction

    def find_contact_points(
        self,
        robot_points: torch.Tensor,
        world_config: WorldConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find the contact points on the robot.

        Args:
            robot_points: The robot points in GLOBAL coordinates.
            world_config: The world configuration.

        Returns:
            in_contact: A boolean tensor of shape (B, n_pts, 1) indicating whether the points are in contact with the terrain.
            dh_points: The penetration depth of the points. Shape (B, n_pts, 1).
        """
        z_points = interpolate_grid(world_config.z_grid, robot_points[..., :2], world_config.max_coord)
        n = surface_normals_from_grads(world_config.z_grid_grad, robot_points[..., :2], world_config.max_coord)
        dh_points = (robot_points[..., 2:3] - z_points) * n[..., 2:3]  # penetration depth as a signed distance from the tangent plane
        in_contact = 0.5 * (1 + torch.tanh((-dh_points / self.config.soft_contact_sigma * (3**0.5))))  # shape (B, n_pts, 1)
        return in_contact, dh_points * in_contact, n

    def update_state(self, state: PhysicsState, dstate: PhysicsStateDer) -> PhysicsState:
        """
        Integrates the states of the rigid body for the next time step.
        """
        # basic kinematics
        next_xd = state.xd + dstate.xdd * self.config.dt
        next_x = state.x + next_xd * self.config.dt
        # joint kinematics
        next_thetas = state.thetas + dstate.thetas_d * self.config.dt
        next_thetas.clamp_(self.robot_model.joint_limits[0], self.robot_model.joint_limits[1])
        # rotation kinematics
        next_omega = state.omega + dstate.omega_d * self.config.dt
        next_q = integrate_quaternion(state.q, next_omega, self.config.dt)
        return PhysicsState(
            x=next_x,
            xd=next_xd,
            q=next_q,
            omega=next_omega,
            thetas=next_thetas,
            batch_size=[self.config.num_robots],
        )

    def assemble_and_transform_robot(
        self, state: PhysicsState, controls: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1.  Prepare joint rotations and construct the driving parts in the robot's local frame
        rots = rot_Y(state.thetas.view(-1, 1)).view(
            self.config.num_robots, self.robot_model.num_driving_parts, 1, 3, 3
        )  # shape (B, n_joints, 1, 3, 3)
        # All of these are expressed in the robot's local frame.
        # The joint points are first rotated in their own frame and then translated to the robot's frame.
        rot_driving_part_pts = torch.matmul(
            rots,
            self.robot_model.joint_local_driving_part_pts.view(
                1,
                self.robot_model.num_driving_parts,
                self.robot_model.points_per_driving_part,
                3,
                1,
            ),  # shape (1, n_joints, n_pts, 3, 1)
        ) + self.robot_model.joint_positions.view(
            1, self.robot_model.num_driving_parts, 1, 3, 1
        )  # shape (B, n_joints, n_pts, 3, 1), the driving part ports for all robots in the robot's local frame
        rot_driving_part_cogs = torch.matmul(
            rots,
            self.robot_model.joint_local_driving_part_cogs.view(1, self.robot_model.num_driving_parts, 1, 3, 1),
        ) + self.robot_model.joint_positions.view(
            1, self.robot_model.num_driving_parts, 1, 3, 1
        )  # shape (B, n_joints, 1, 3, 1), the cog positions of the driving parts in the robot's local frame
        rot_driving_part_inertias = torch.matmul(
            torch.matmul(
                rots,
                self.robot_model.driving_part_inertias.view(1, self.robot_model.num_driving_parts, 1, 3, 3),
            ),
            rots.transpose(-1, -2),
        )  # shape (B, n_joints, 1, 3, 3)
        # 2. Add the body to compute the overall inertia and CoG
        cog_overall = torch.sum(
            rot_driving_part_cogs * self.robot_model.driving_part_masses.view(1, self.robot_model.num_driving_parts, 1, 1, 1),
            dim=1,
        ) + (self.robot_model.body_mass * self.robot_model.body_cog.view(1, 3, 1)).unsqueeze(1)  # shape (B, 1, 3, 1)
        cog_overall = cog_overall / self.robot_model.total_mass  # shape (B, 1, 3, 1)
        # Compute vectors from overall CoG to each part's CoG
        d_driving = rot_driving_part_cogs - cog_overall.unsqueeze(1)  # shape (B, n_joints, 1, 3, 1)
        d_body = self.robot_model.body_cog.view(1, 1, 3, 1) - cog_overall  # shape (B, 1, 3, 1)
        # Compute translation terms for driving parts
        d_driving_sq = torch.sum(d_driving**2, dim=-2, keepdim=True)  # shape (B, n_joints, 1, 1, 1)
        translation_term_driving = d_driving_sq * self._I_3x3 - torch.matmul(d_driving, d_driving.transpose(-1, -2))  # shape (B, n_joints, 1, 3, 3)
        translation_term_driving = translation_term_driving * self.robot_model.driving_part_masses.view(
                                                                  1, self.robot_model.num_driving_parts, 1, 1, 1
                                                              )  # shape (B, n_joints, 1, 3, 3)
        # Compute translation term for the body
        d_body_sq = torch.sum(d_body**2, dim=-2, keepdim=True)  # shape (B, 1, 1, 1)
        translation_term_body = d_body_sq * self._I_3x3.squeeze(1) - torch.matmul(d_body, d_body.transpose(-1, -2))  # shape (B, 1, 3, 3)
        translation_term_body = translation_term_body * self.robot_model.body_mass  # shape (B, 1, 3, 3)
        # Compute total inertia tensor with respect to overall CoG
        I_overall = torch.sum(rot_driving_part_inertias + translation_term_driving, dim=1) + (
            self.robot_model.body_inertia.unsqueeze(0) + translation_term_body
        )  # shape (B, 1, 3, 3)
        # 3. Transform everything to the world frame
        R_world = q_to_R(state.q)  # shape (B, 3, 3)
        t_world = state.x.unsqueeze(1)  # shape (B, 1, 3)
        cog_overall_world = torch.matmul(R_world.unsqueeze(1), cog_overall) + t_world.view(self.config.num_robots, 1, 3, 1)  # shape (B, 1, 3, 1)
        I_overall_world = torch.matmul(
            torch.matmul(R_world, I_overall.squeeze(1)),
            R_world.transpose(-1, -2),  # matmul result shape (B, 3, 3)
        )  # shape (B, 3, 3)
        driving_parts_world = torch.matmul(R_world.view(self.config.num_robots, 1, 1, 3, 3), rot_driving_part_pts) + t_world.view(
            self.config.num_robots, 1, 1, 3, 1
        )  # shape (B, n_joints, n_pts, 3, 1)
        body_world = torch.matmul(R_world.unsqueeze(1), self.robot_model.body_points.view(1, -1, 3, 1)) + t_world.view(
            self.config.num_robots, 1, 3, 1
        )  # shape (B, n_pts, 3, 1)
        thrust_directions_world = torch.matmul(
            torch.matmul(R_world.view(self.config.num_robots, 1, 1, 3, 3), rots),  # shape  (B, n_joints, 1, 3, 3)
            self.robot_model.thrust_directions.view(
                1,
                self.robot_model.num_driving_parts,
                self.robot_model.points_per_driving_part,
                3,
                1,
            ),
        )
        # 4. Concatenate all points
        robot_points = torch.cat((driving_parts_world.view(self.config.num_robots, -1, 3), body_world.squeeze(-1)), dim=1)
        # 5. Compute all thrust directions scaled by commanded velocity
        velocity_cmd = controls[:, : self.robot_model.num_driving_parts].clamp(-self.robot_model.v_max, self.robot_model.v_max)  # shape (B, n_joints)
        thrust_directions_world = thrust_directions_world * velocity_cmd.view(
            self.config.num_robots, self.robot_model.num_driving_parts, 1, 1, 1
        )  # shape (B, n_joints, 1, 3, 1)
        thrust_vectors = torch.cat(
            (
                thrust_directions_world.view(self.config.num_robots, -1, 3),
                torch.zeros_like(body_world.squeeze(-1)),
            ),
            dim=1,
        )
        return robot_points, thrust_vectors, cog_overall_world.squeeze(-1), I_overall_world
