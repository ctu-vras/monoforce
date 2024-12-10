import torch
from torchdiffeq import odeint, odeint_adjoint
from ..dphys_config import DPhysConfig


def normalized(x, eps=1e-6, dim=-1):
    """
    Normalizes the input tensor.

    Parameters:
    - x: Input tensor.
    - eps: Small value to avoid division by zero.

    Returns:
    - Normalized tensor.
    """
    norm = torch.norm(x, dim=dim, keepdim=True)
    return x / torch.clamp(norm, min=eps)


def skew_symmetric(v):
    """
    Returns the skew-symmetric matrix of a vector.

    Parameters:
    - v: Input vector.

    Returns:
    - Skew-symmetric matrix of the input vector.
    """
    assert v.dim() == 2 and v.shape[1] == 3
    U = torch.zeros(v.shape[0], 3, 3, device=v.device)
    U[:, 0, 1] = -v[:, 2]
    U[:, 0, 2] = v[:, 1]
    U[:, 1, 2] = -v[:, 0]
    U[:, 1, 0] = v[:, 2]
    U[:, 2, 0] = -v[:, 1]
    U[:, 2, 1] = v[:, 0]
    return U

def generate_control_inputs(n_trajs=10,
                            time_horizon=5.0, dt=0.01,
                            v_range=(-1.0, 1.0), w_range=(-1.0, 1.0)):
    """
    Generates control inputs for the robot trajectories.

    Parameters:
    - n_trajs: Number of trajectories.
    - time_horizon: Time horizon for each trajectory.
    - dt: Time step.
    - v_range: Range of the forward speed.
    - w_range: Range of the rotational speed.

    Returns:
    - Linear and angular velocities for the robot trajectories: (n_trajs, time_steps, 2).
    - Time stamps for the trajectories.
    """
    N = int(time_horizon / dt)
    time_stamps = torch.linspace(0, time_horizon, N)

    v = torch.rand(n_trajs) * (v_range[1] - v_range[0]) + v_range[0]  # Forward speed
    w = torch.rand(n_trajs) * (w_range[1] - w_range[0]) + w_range[0]  # Rotational speed

    # repeat the control inputs for each time step
    v = v.unsqueeze(1).repeat(1, N)
    w = w.unsqueeze(1).repeat(1, N)

    # stack the control inputs
    control_inputs = torch.stack([v, w], dim=-1)

    return control_inputs, time_stamps


def vw_to_track_vels(v, w, robot_size, n_tracks):
    """
    Converts the forward and rotational speeds to track velocities.

    Parameters:
    - v: Forward speed.
    - w: Rotational speed.
    - robot_size: Size of the robot.
    - n_tracks: Number of tracks (2 or 4).

    Returns:
    - Track velocities.
    """
    Lx, Ly = robot_size
    if n_tracks == 2:
        v_L = v - w * (Ly / 2.0)  # Left wheel velocity
        v_R = v + w * (Ly / 2.0)  # Right wheel velocity
        # left, right
        track_vels = torch.stack([v_L, v_R], dim=-1)
    elif n_tracks == 4:
        # front left, front right, rear left, rear right
        v_FL = v - w * Ly / 2.0
        v_FR = v + w * Ly / 2.0
        v_RL = v - w * Ly / 2.0
        v_RR = v + w * Ly / 2.0
        track_vels = torch.stack([v_FL, v_FR, v_RL, v_RR], dim=-1)
    else:
        raise ValueError('n_tracks must be 2 or 4')

    return track_vels


class DPhysics(torch.nn.Module):
    def __init__(self, dphys_cfg=DPhysConfig(), device='cpu'):
        super(DPhysics, self).__init__()
        self.dphys_cfg = dphys_cfg
        self.device = device
        self.I = self.dphys_cfg.robot_I.to(self.device)  # 3x3 inertia tensor, kg*m^2
        self.I_inv = torch.inverse(self.I)  # inverse of the inertia tensor
        self.x_points = self.dphys_cfg.robot_points.to(self.device)  # robot body points
        
        self.z_grid = None
        self.friction = None
        self.stiffness = None
        self.damping = None
        self.controls = None
        T, dt = self.dphys_cfg.traj_sim_time, self.dphys_cfg.dt
        self.ts = torch.linspace(0, T, int(T / dt)).to(self.device)

        self.integrator = self.dynamics_odeint if self.dphys_cfg.use_odeint else self.dynamics

    def forward_kinematics(self, t, state):
        # unpack state
        x, xd, R, omega = state
        assert x.dim() == 2 and x.shape[1] == 3  # (B, 3)
        assert xd.dim() == 2 and xd.shape[1] == 3  # (B, 3)
        assert R.dim() == 3 and R.shape[-2:] == (3, 3)  # (B, 3, 3)
        assert omega.dim() == 2 and omega.shape[1] == 3  # (B, 3)

        x_points = self.x_points @ R.transpose(1, 2) + x.unsqueeze(1)
        assert x_points.dim() == 3 and x_points.shape[-1] == 3  # (B, N, 3)
        B, n_pts, D = x_points.shape

        # motion of point composed of cog motion and rotation of the rigid body
        # Koenig's theorem in mechanics: v_i = v_cog + omega x (r_i - r_cog)
        xd_points = xd.unsqueeze(1) + torch.linalg.cross(omega.unsqueeze(1), x_points - x.unsqueeze(1))
        assert xd_points.shape == (B, n_pts, 3)

        for p in self.dphys_cfg.driving_parts:
            assert p.dim() == 1 and p.shape[0] == x_points.shape[1]  # (N,)

        # closest time step in the control inputs
        t_id = torch.argmin(torch.abs(t - self.ts))
        controls_t = self.controls[:, t_id]
        assert controls_t.dim() == 2 and controls_t.shape[0] == x.shape[0]  # (B, 2)
        assert controls_t.shape[1] == 2  # linear and angular velocities
        assert self.z_grid.dim() == 3  # (B, H, W)

        # compute the terrain properties at the robot points
        z_points, n = self.interpolate_grid(self.z_grid, x_points[..., 0], x_points[..., 1], return_normals=True)
        z_points = z_points.unsqueeze(-1)
        assert z_points.shape == (B, n_pts, 1)
        assert n.shape == (B, n_pts, 3)

        stiffness_points = self.interpolate_grid(self.stiffness, x_points[..., 0], x_points[..., 1]).unsqueeze(-1)
        assert stiffness_points.shape == (B, n_pts, 1)
        damping_points = self.interpolate_grid(self.damping, x_points[..., 0], x_points[..., 1]).unsqueeze(-1)
        assert damping_points.shape == (B, n_pts, 1)
        friction_points = self.interpolate_grid(self.friction, x_points[..., 0], x_points[..., 1]).unsqueeze(-1)
        assert friction_points.shape == (B, n_pts, 1)

        # check if the rigid body is in contact with the terrain
        dh_points = x_points[..., 2:3] - z_points
        # in_contact = dh_points < 0.
        # soft contact model
        in_contact = torch.sigmoid(-10. * dh_points)
        assert in_contact.shape == (B, n_pts, 1)

        # reaction at the contact points as spring-damper forces
        m, g = self.dphys_cfg.robot_mass, self.dphys_cfg.gravity
        xd_points_n = (xd_points * n).sum(dim=2).unsqueeze(2)  # normal velocity
        assert xd_points_n.shape == (B, n_pts, 1)
        F_spring = -torch.mul((stiffness_points * dh_points + damping_points * xd_points_n), n)  # F_s = -k * dh - b * v_n
        F_spring = torch.mul(F_spring, in_contact) / n_pts  # apply forces only at the contact points
        F_spring = torch.clamp(F_spring, min=-m*g, max=m*g)
        assert F_spring.shape == (B, n_pts, 3)

        # static and dynamic friction forces: https://en.wikipedia.org/wiki/Friction
        thrust_dir = normalized(R[..., 0])  # direction of the thrust
        N = torch.norm(F_spring, dim=2)  # normal force magnitude at the contact points
        track_vels = vw_to_track_vels(v=controls_t[:, 0], w=controls_t[:, 1],
                                      robot_size=self.dphys_cfg.robot_size, n_tracks=len(self.dphys_cfg.driving_parts))
        assert track_vels.shape == (B, len(self.dphys_cfg.driving_parts))
        cmd_vels = torch.zeros_like(xd_points)
        for i in range(len(self.dphys_cfg.driving_parts)):
            mask = self.dphys_cfg.driving_parts[i]
            u = track_vels[:, i].unsqueeze(1) * thrust_dir
            cmd_vels[:, mask] = u.unsqueeze(1)
        mu_cmd_vels = friction_points * cmd_vels
        mu_cmd_vels = torch.clamp(mu_cmd_vels, min=-self.dphys_cfg.vel_max, max=self.dphys_cfg.vel_max)
        vels_diff = mu_cmd_vels - xd_points
        vels_diff_n = (vels_diff * n).sum(dim=2).unsqueeze(2)  # normal velocity difference
        vels_diff_tau = vels_diff - vels_diff_n * n  # tangential velocity difference
        F_friction = N.unsqueeze(2) * vels_diff_tau  # F_f = mu * N * v
        # F_friction = N.unsqueeze(2) * vels_diff
        F_friction = torch.clamp(F_friction, min=-m*g, max=m*g)
        assert F_friction.shape == (B, n_pts, 3)

        # rigid body rotation: M = sum(r_i x F_i)
        torque = torch.sum(torch.linalg.cross(x_points - x.unsqueeze(1), F_spring + F_friction), dim=1)
        omega_d = torque @ self.I_inv.transpose(0, 1)  # omega_d = I^(-1) M
        omega_d = torch.clamp(omega_d, min=-self.dphys_cfg.omega_max, max=self.dphys_cfg.omega_max)
        Omega_skew = skew_symmetric(omega)  # Omega_skew = [omega]_x
        dR = Omega_skew @ R  # dR = [omega]_x R
        assert omega_d.shape == (B, 3)
        assert dR.shape == (B, 3, 3)

        # motion of the cog
        F_grav = m * g * torch.as_tensor(self.dphys_cfg.gravity_direction, device=self.device).unsqueeze(0)  # F_grav = [0, 0, -m * g]
        F_cog = F_grav + F_spring.sum(dim=1) + F_friction.sum(dim=1)  # ma = sum(F_i)
        xdd = F_cog / m  # a = F / m
        assert xdd.shape == (B, 3)

        dstate = (xd, xdd, dR, omega_d)
        forces = (F_spring, F_friction)

        return dstate, forces

    def update_state(self, state, dstate, dt):
        """
        Integrates the states of the rigid body for the next time step.
        """
        x, xd, R, omega = state
        _, xdd, dR, omega_d = dstate

        xd = self.integration_step(xd, xdd, dt, mode=self.dphys_cfg.integration_mode)
        x = self.integration_step(x, xd, dt, mode=self.dphys_cfg.integration_mode)
        omega = self.integration_step(omega, omega_d, dt, mode=self.dphys_cfg.integration_mode)
        # R = self.integration_step(R, dR, dt, mode=self.dphys_cfg.integration_mode)
        R = self.integrate_rotation(R, omega, dt)

        state = (x, xd, R, omega)

        return state

    @staticmethod
    def integrate_rotation(R, omega, dt, eps=1e-6):
        """
        Integrates the rotation matrix for the next time step using Rodrigues' formula.

        Parameters:
        - R: Tensor of rotation matrices.
        - omega: Tensor of angular velocities.
        - dt: Time step.
        - eps: Small value to avoid division by zero.

        Returns:
        - Updated rotation matrices.

        Reference:
            https://math.stackexchange.com/questions/167880/calculating-new-rotation-matrix-with-its-derivative-given
        """
        assert R.dim() == 3 and R.shape[-2:] == (3, 3)
        assert omega.dim() == 2 and omega.shape[1] == 3
        assert dt > 0

        # Compute the skew-symmetric matrix of the angular velocities
        Omega_x = skew_symmetric(omega)

        # Compute exponential map of the skew-symmetric matrix
        theta = torch.norm(omega, dim=-1, keepdim=True).unsqueeze(-1)

        # Normalize the angular velocities
        Omega_x_norm = Omega_x / torch.clamp(theta, min=eps)

        # Rodrigues' formula: R_new = R * (I + Omega_x * sin(theta * dt) + Omega_x^2 * (1 - cos(theta * dt)))
        I = torch.eye(3).to(R.device)
        R_new = R @ (I + Omega_x_norm * torch.sin(theta * dt) + Omega_x_norm @ Omega_x_norm * (1 - torch.cos(theta * dt)))

        return R_new

    @staticmethod
    def integration_step(x, xd, dt, mode='euler'):
        """
        Performs an integration step using the Euler method.

        Parameters:
        - x: Tensor of positions.
        - xd: Tensor of velocities.
        - dt: Time step.

        Returns:
        - Updated positions and velocities.
        """
        if mode == 'euler':
            x = x + xd * dt
        elif mode == 'rk4':
            k1 = dt * xd
            k2 = dt * (xd + k1 / 2)
            k3 = dt * (xd + k2 / 2)
            k4 = dt * (xd + k3)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            raise ValueError(f'Unknown integration mode: {mode}')
        return x

    def interpolate_grid(self, grid, x_query, y_query, return_normals=False):
        """
        Interpolates the height at the desired (x_query, y_query) coordinates.

        Parameters:
        - grid: Tensor of grid values corresponding to the x and y coordinates (3D array), (B, H, W).
        - x_query: Tensor of desired x coordinates for interpolation (2D array), (B, N).
        - y_query: Tensor of desired y coordinates for interpolation (2D array), (B, N).
        - return_normals: Bool to return the surface normals at the queried coordinates or not.

        Returns:
        - Interpolated grid values at the queried coordinates, (B, N).
        - Surface normals at the queried coordinates (if return_normals=True), (B, N, 3).
        """
        # unpack config
        d_max = self.dphys_cfg.d_max
        grid_res = self.dphys_cfg.grid_res

        # Ensure inputs are tensors
        grid = torch.as_tensor(grid)
        x_query = torch.as_tensor(x_query)
        y_query = torch.as_tensor(y_query)

        # Get the grid dimensions
        B, H, W = grid.shape

        # Flatten the grid coordinates
        z_grid_flat = grid.reshape(B, -1)

        # Flatten the query coordinates
        x_query_flat = x_query.reshape(B, -1)
        y_query_flat = y_query.reshape(B, -1)

        # Compute the indices of the grid points surrounding the query points
        x_i = ((x_query_flat + d_max) / grid_res).long()
        y_i = ((y_query_flat + d_max) / grid_res).long()

        # Compute the fractional part of the indices
        x_frac = (x_query_flat + d_max) / grid_res - x_i.float()
        y_frac = (y_query_flat + d_max) / grid_res - y_i.float()

        # Compute the indices of the grid points
        i_c = y_i + H * x_i
        i_f = y_i + H * (x_i + 1)
        i_l = (y_i + 1) + H * x_i
        i_fl = (y_i + 1) + H * (x_i + 1)
        # Clamp the indices to avoid out-of-bound errors
        i_c = torch.clamp(i_c, 0, H * W - 1)
        i_f = torch.clamp(i_f, 0, H * W - 1)
        i_l = torch.clamp(i_l, 0, H * W - 1)
        i_fl = torch.clamp(i_fl, 0, H * W - 1)

        # Interpolate the z values (linear interpolation)
        z_center = z_grid_flat.gather(1, i_c)
        z_front = z_grid_flat.gather(1, i_f)
        z_left = z_grid_flat.gather(1, i_l)
        z_front_left = z_grid_flat.gather(1, i_fl)
        z_query = (1 - x_frac) * (1 - y_frac) * z_center + \
                  (1 - x_frac) * y_frac * z_front + \
                  x_frac * (1 - y_frac) * z_left + \
                  x_frac * y_frac * z_front_left

        if return_normals:
            # Estimate normals using the height map
            dz_dx = (z_front - z_center) / grid_res
            dz_dy = (z_left - z_center) / grid_res
            n = torch.stack([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=-1)
            n = normalized(n)
            return z_query, n

        return z_query

    def forward_kinematics_extended_state(self, t, state_extended):
        """
        Extended forward kinematics function that takes the extended state as input.
        """
        x, xd, R, omega, F_spring, F_friction = state_extended
        state = (x, xd, R, omega)
        dstate, forces = self.forward_kinematics(t, state)
        dstate = dstate + forces
        return dstate

    def dynamics(self, state):
        Xs, Xds, Rs, Omegas, F_springs, F_frictions = [], [], [], [], [], []
        for t in self.ts:
            # forward kinematics
            dstate, forces = self.forward_kinematics(t=t, state=state)
            # update state: integration steps
            state = self.update_state(state, dstate, self.dphys_cfg.dt)

            # unpack state, its differential, and forces
            x, xd, R, omega = state
            F_spring, F_friction = forces

            # save states
            Xs.append(x)
            Xds.append(xd)
            Rs.append(R)
            Omegas.append(omega)

            # save forces
            F_springs.append(F_spring)
            F_frictions.append(F_friction)

        # stack the states and forces
        Xs = torch.stack(Xs, dim=1)
        Xds = torch.stack(Xds, dim=1)
        Rs = torch.stack(Rs, dim=1)
        Omegas = torch.stack(Omegas, dim=1)
        F_springs = torch.stack(F_springs, dim=1)
        F_frictions = torch.stack(F_frictions, dim=1)

        return Xs, Xds, Rs, Omegas, F_springs, F_frictions

    def dynamics_odeint(self, state):
        """
        Simulates the dynamics of the robot using ODE solver.
        """
        B = state[0].shape[0]
        N_pts = len(self.x_points)
        N_ts = len(self.ts)
        f_spring = torch.zeros(B, N_pts, 3, device=self.device)
        f_friction = torch.zeros(B, N_pts, 3, device=self.device)
        forces = (f_spring, f_friction)
        state_extended = tuple(state) + tuple(forces)
        state_extended = odeint(self.forward_kinematics_extended_state, state_extended, self.ts,
                                method=self.dphys_cfg.integration_mode, rtol=1e-3, atol=1e-3)
        Xs, Xds, Rs, Omegas, F_springs, F_frictions = state_extended

        # transpose the states and forces to (B, N, D)
        Xs = Xs.permute(1, 0, 2)
        assert Xs.shape == (B, N_ts, 3)
        Xds = Xds.permute(1, 0, 2)
        assert Xds.shape == (B, N_ts, 3)
        Rs = Rs.permute(1, 0, 2, 3)
        assert Rs.shape == (B, N_ts, 3, 3)
        Omegas = Omegas.permute(1, 0, 2)
        assert Omegas.shape == (B, N_ts, 3)
        F_springs = F_springs.permute(1, 0, 2, 3)
        assert F_springs.shape == (B, N_ts, N_pts, 3)
        F_frictions = F_frictions.permute(1, 0, 2, 3)
        assert F_frictions.shape == (B, N_ts, N_pts, 3)

        return Xs, Xds, Rs, Omegas, F_springs, F_frictions

    def dphysics(self, z_grid, controls, state=None, stiffness=None, damping=None, friction=None):
        """
        Simulates the dynamics of the robot moving on the terrain.

        Parameters:
        - z_grid: Tensor of the height map (B, H, W).
        - controls: Tensor of control inputs (B, N, 2).
        - state: Tuple of the robot state (x, xd, R, omega).
        - stiffness: scalar or Tensor of the stiffness values at the robot points (B, H, W).
        - damping: scalar or Tensor of the damping values at the robot points (B, H, W).
        - friction: scalar or Tensor of the friction values at the robot points (B, H, W).

        Returns:
        - Tuple of the robot states and forces:
            - states: Tuple of the robot states (x, xd, R, omega).
            - forces: Tuple of the forces (F_springs, F_frictions).
        """
        # unpack config
        dt = self.dphys_cfg.dt
        T = self.dphys_cfg.traj_sim_time
        batch_size = z_grid.shape[0]

        # initial state
        if state is None:
            x = torch.tensor([0.0, 0.0, 0.0]).to(self.device).repeat(batch_size, 1)
            xd = torch.zeros_like(x); xd[:, 0] = controls[:, 0, 0]  # initial forward speed
            R = torch.eye(3).to(self.device).repeat(batch_size, 1, 1)
            omega = torch.zeros_like(x); omega[:, 2] = controls[:, 0, 1]  # initial rotational speed
            state = (x, xd, R, omega)

        # terrain properties
        stiffness = self.dphys_cfg.stiffness.repeat(batch_size, 1, 1) if stiffness is None else stiffness
        damping = self.dphys_cfg.damping.repeat(batch_size, 1, 1) if damping is None else damping
        friction = self.dphys_cfg.friction.repeat(batch_size, 1, 1) if friction is None else friction
        self.z_grid = z_grid.to(self.device)
        self.stiffness = stiffness.to(self.device)
        self.damping = damping.to(self.device)
        self.friction = friction.to(self.device)

        # start robot at the terrain height (not under or above the terrain)
        x = state[0]
        x_points = self.x_points.repeat(batch_size, 1, 1)
        x_points = x_points @ state[2].transpose(1, 2) + x.unsqueeze(1)
        z_interp = self.interpolate_grid(self.z_grid, x_points[..., 0], x_points[..., 1]).mean(dim=1, keepdim=True)
        x[..., 2:3] = z_interp

        N_ts = min(int(T / dt), controls.shape[1])
        B = state[0].shape[0]
        assert controls.shape == (B, N_ts, 2), f'Its shape {controls.shape} != {(B, N_ts, 2)}'  # (B, N, 2), v, w
        self.controls = controls
        self.ts = self.ts[:N_ts]

        # dynamics of the rigid body
        Xs, Xds, Rs, Omegas, F_springs, F_frictions = self.integrator(state)

        # mg = k * delta_h, at equilibrium, delta_h = mg / k
        delta_h = self.dphys_cfg.robot_mass * self.dphys_cfg.gravity / self.stiffness.mean()
        Xs[..., 2] = Xs[..., 2] + delta_h.abs()  # add the equilibrium height

        States = Xs, Xds, Rs, Omegas
        Forces = F_springs, F_frictions

        return States, Forces

    def forward(self, z_grid, controls, state=None, vis=False, stiffness=None, damping=None, friction=None):
        states, forces = self.dphysics(z_grid, controls, state,
                                       stiffness=stiffness, damping=damping, friction=friction)
        if vis:
            with torch.no_grad():
                self.visualize(states, forces, z_grid, friction=friction)
        return states, forces

    def visualize(self, states, forces, z_grid, friction=None, step=10, batch_i=0):
        # visualize using mayavi
        from monoforce.vis import setup_visualization, animate_trajectory

        xs, xds, rs, omegas = [s[batch_i].cpu().numpy() for s in states]
        F_spring, F_friction = [f[batch_i].cpu().numpy() for f in forces]
        x_grid_np, y_grid_np = self.dphys_cfg.x_grid.cpu().numpy(), self.dphys_cfg.y_grid.cpu().numpy()
        z_grid_np = z_grid[batch_i].cpu().numpy()
        if friction is not None:
            friction_np = friction[batch_i].cpu().numpy()
        else:
            friction_np = self.dphys_cfg.friction.cpu().numpy()
        x_points = self.dphys_cfg.robot_points.cpu().numpy()

        # set up the visualization
        vis_cfg = setup_visualization(states=(xs, xds, rs, omegas),
                                      x_points=x_points,
                                      forces=(F_spring, F_friction),
                                      x_grid=x_grid_np, y_grid=y_grid_np, z_grid=z_grid_np)

        # visualize animated trajectory
        animate_trajectory(states=(xs, xds, rs, omegas),
                           x_points=x_points,
                           forces=(F_spring, F_friction),
                           z_grid=z_grid_np,
                           friction=friction_np,
                           vis_cfg=vis_cfg, step=step)
