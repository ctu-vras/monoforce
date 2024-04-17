import warnings
import torch
from torch import nn
import numpy as np
from torchdiffeq import odeint, odeint_adjoint
from ..transformations import rot2rpy
from ..utils import skew_symmetric
from ..config import DPhysConfig
from ..control import pose_control


torch.set_default_dtype(torch.float32)

class State:
    def __init__(self,
                 xyz=torch.zeros((3, 1)),
                 rot=torch.eye(3),
                 vel=torch.zeros((3, 1)),
                 omega=torch.zeros((3, 1)),
                 forces=torch.zeros((3, 10)), device='cpu'):
        """
        pos: 3x1, x,y,z position
        rot: 3x3, rotation matrix
        vel: 3x1, linear velocity
        omega: 3x1, angular velocity
        forces: 3xN, N is the number of contact points
        """
        self.xyz = torch.as_tensor(xyz).view(3, 1).to(device)
        self.rot = torch.as_tensor(rot).view(3, 3).to(device)
        self.vel = torch.as_tensor(vel).view(3, 1).to(device)
        self.omega = torch.as_tensor(omega).view(3, 1).to(device)
        self.forces = torch.as_tensor(forces).view(3, -1).to(device)

    def as_tuple(self):
        state = (self.xyz, self.rot, self.vel, self.omega, self.forces)
        return state

    def from_tuple(self, state):
        self.xyz = state[0]
        self.rot = state[1]
        self.vel = state[2]
        self.omega = state[3]
        self.forces = state[4]

    def clone(self):
        return State(self.xyz.clone(), self.vel.clone(), self.rot.clone(), self.omega.clone(), self.forces.clone())

    def to(self, device):
        self.xyz = self.xyz.to(device)
        self.rot = self.rot.to(device)
        self.vel = self.vel.to(device)
        self.omega = self.omega.to(device)
        self.forces = self.forces.to(device)

    def __getitem__(self, i):
        return self.as_tuple()[i]

    def __str__(self):
        return str(self.__dict__)

    def update(self, dstate, dt, inplace=False):
        assert isinstance(dstate, tuple)
        assert len(dstate) == 5

        dpos_x, dpos_R, dvel_x, dvel_omega, dforces = dstate

        st3 = self.omega + dvel_omega.view(3, 1) * dt
        vel_omega_skew = skew_symmetric(st3)

        c1 = torch.sin(torch.as_tensor(dt))
        c2 = (1 - torch.cos(torch.as_tensor(dt)))

        state = (self.xyz + dpos_x * dt,
                 (torch.eye(3, device=vel_omega_skew.device) + c1 * vel_omega_skew + c2 * (vel_omega_skew ** 2)) @ self.rot,
                 self.vel + dvel_x.view(3, 1) * dt,
                 self.omega + dvel_omega.view(3, 1) * dt,
                 dforces)

        if inplace:
            self.from_tuple(state)
        else:
            return State(*state, device=vel_omega_skew.device)


class RigidBodySoftTerrain(nn.Module):

    def __init__(self,
                 height=np.zeros([8, 3]),
                 grid_res=0.1,
                 damping=10.0, elasticity=10.0, friction=0.9,
                 mass=10.0, gravity=9.8,
                 inertia=5.0 * np.eye(3),
                 state=State(),
                 vel_tracks=np.zeros(2),
                 adjoint=False,
                 device=torch.device('cpu'),
                 use_ode=False,
                 soft_layer_height=0.2,
                 interaction_model='diffdrive',
                 Kp_rho=2.0,  # position proportional gain
                 Kp_theta=50.0,  # heading proportional gain
                 Kp_yaw=1.0,  # yaw (at a pose) proportional gain
                 learn_height=True,
                 robot_model='husky'
                 ):
        super().__init__()
        self.device = device
        self.gravity = nn.Parameter(torch.as_tensor([gravity]))
        self.use_ode = use_ode
        self.grid_res = grid_res
        self.t0 = nn.Parameter(torch.tensor([0.0]))
        self.height = torch.as_tensor(height, device=self.device)
        self.height0 = self.height.clone()
        self.height_soft = torch.ones_like(self.height, device=self.device) * soft_layer_height
        self.height = nn.Parameter(self.height) if learn_height else self.height
        self.height_soft = nn.Parameter(self.height_soft)
        self.damping = nn.Parameter(torch.ones_like(self.height) * damping)
        self.elasticity = nn.Parameter(torch.ones_like(self.height) * elasticity)
        self.friction = nn.Parameter(torch.ones_like(self.height) * friction)
        self.elasticity_rigid = 2000.
        self.damping_rigid = 200.

        self.state = state

        self.robot_points = nn.Parameter(self.create_robot_model(robot_model))
        self.init_forces = nn.Parameter(torch.zeros_like(self.robot_points))

        self.mass = nn.Parameter(torch.tensor([mass]))
        self.inertia = torch.tensor(inertia, dtype=self.height.dtype, device=self.device)
        self.inertia_inv = torch.inverse(self.inertia)
        # self.vel_tracks = nn.Parameter(torch.tensor(vel_tracks, device=self.device))
        self.vel_tracks = torch.tensor(vel_tracks, device=self.device)

        self.odeint = odeint_adjoint if adjoint else odeint

        self.pos_x = None
        self.pos_R = None
        self.vel_x = None
        self.vel_omega = None
        self.forces = None

        self.interaction_model = interaction_model

        # controller parameters (path follower)
        self.Kp_rho = nn.Parameter(torch.tensor([Kp_rho], device=self.device))
        self.Kp_theta = nn.Parameter(torch.tensor([Kp_theta], device=self.device))
        self.Kp_yaw = nn.Parameter(torch.tensor([Kp_yaw], device=self.device))

    def create_robot_model(self, model='husky'):
        if model == 'tradr':
            size = (1.0, 0.5)
            s_x, s_y = size
            n_pts = 10
            px = torch.hstack([torch.linspace(-s_x/2., s_x/2., n_pts//2), torch.linspace(-s_x/2., s_x/2., n_pts//2)])
            py = torch.hstack([s_y/2. * torch.ones(n_pts//2), -s_y/2. * torch.ones(n_pts//2)])
            pz = torch.hstack([torch.tensor([0.2, 0.1, 0.0, 0.0, 0.0]), torch.tensor([0.2, 0.1, 0.0, 0.0, 0.0])])
        elif model == 'husky':
            size = (0.9, 0.6)
            s_x, s_y = size
            n_pts = 10
            px = torch.hstack([torch.linspace(-s_x/2., s_x/2., n_pts//2), torch.linspace(-s_x/2., s_x/2., n_pts//2)])
            py = torch.hstack([s_y / 2. * torch.ones(n_pts // 2), -s_y / 2. * torch.ones(n_pts // 2)])
            pz = torch.zeros(n_pts)
        elif model == 'marv':
            raise NotImplementedError
        elif model == 'warthog':
            raise NotImplementedError
        else:
            print(f'Unknown robot model: {model}')
            raise NotImplementedError
        robot_points = torch.stack((px, py, pz)).to(self.device)
        return robot_points

    def forward(self, t, state):
        """
        Forward pass of the robot-terrain interaction model.
        @param t: time moments
        @param state: state vector
        @return: state derivative
        """
        if self.interaction_model == 'omni':
            return self.forward_omni(t, state)
        elif self.interaction_model == 'diffdrive':
            return self.forward_diffdrive(t, state)
        elif self.interaction_model == 'rigid_layer':
            return self.forward_rigid_layer(t, state)
        elif self.interaction_model == 'rigid_soft_layers':
            return self.forward_rigid_soft_layers(t, state)
        else:
            raise NotImplementedError

    def forward_omni(self, t, state):
        pos_x, pos_R, vel_x, vel_omega, f_old = state

        dpos_x = vel_x
        vel_omega_skew = skew_symmetric(vel_omega).to(self.device)
        dpos_R = vel_omega_skew @ pos_R
        points = pos_R @ self.robot_points + pos_x
        dpoints = vel_omega_skew @ (points - pos_x) + vel_x

        # interpolate
        H, W = self.height.shape
        xy_grid = points[0:2, :] / self.grid_res + torch.tensor([H / 2., W / 2.], device=self.device).view((2, 1))
        h = self.sample_by_interp(self.height, xy_grid)
        e = self.elasticity_rigid  # self.elasticity[idx_points_x, idx_points_y]  # self.sample_by_interp(self.elasticity, points[0:2, :])
        d = self.damping_rigid  # self.damping[idx_points_x, idx_points_y]  # self.sample_by_interp(self.damping, points[0:2, :])

        # contacts
        # contact = (points[2, :] <= h)
        contact = self.soft_contact(h, points)

        # Compute terrain + gravity forces
        z = torch.tile(torch.tensor([[0], [0], [1]]), (1, points.shape[1])).to(self.device)
        forces = (z * (e * (h - points[2, :]) - d * dpoints[2, :])) * contact
        fg = self.mass * self.gravity  # * (points[2, :] >= h)
        forces[2, :] = forces[2, :] - fg  # * (1 - contact.float())

        # Accelerations: linear and angular accelerations computed from forces
        dvel_x = (forces / self.mass).sum(dim=1)
        dvel_omega = self.inertia_inv @ torch.cross(points - pos_x, forces).sum(dim=1)

        return dpos_x, dpos_R, dvel_x, dvel_omega, forces  # _track #torch.zeros_like(self.f)

    def forward_diffdrive(self, t, state):
        pos_x, pos_R, vel_x, vel_omega, f_old = state

        yaw = rot2rpy(pos_R)[2]
        dpos_x = torch.zeros_like(pos_x)
        dpos_x[0] = vel_x[0] * torch.cos(yaw)
        dpos_x[1] = vel_x[0] * torch.sin(yaw)
        dpos_x[2] = vel_x[2]

        vel_omega_skew = skew_symmetric(vel_omega).to(self.device)
        dpos_R = vel_omega_skew @ pos_R
        points = pos_R @ self.robot_points + pos_x
        dpoints = vel_omega_skew @ (points - pos_x) + vel_x

        # interpolate
        H, W = self.height.shape
        xy_grid = points[0:2, :] / self.grid_res + torch.tensor([H / 2., W / 2.], device=self.device).view((2, 1))
        h = self.sample_by_interp(self.height, xy_grid)
        e = self.elasticity_rigid  # self.elasticity[idx_points_x, idx_points_y]  # self.sample_by_interp(self.elasticity, points[0:2, :])
        d = self.damping_rigid  # self.damping[idx_points_x, idx_points_y]  # self.sample_by_interp(self.damping, points[0:2, :])

        # contacts
        # contact = (points[2, :] <= h)
        contact = self.soft_contact(h, points)

        # Compute terrain + gravity forces
        z = torch.tile(torch.tensor([[0], [0], [1]]), (1, points.shape[1])).to(self.device)
        forces = (z * (e * (h - points[2, :]) - d * dpoints[2, :])) * contact
        fg = self.mass * self.gravity  # * (points[2, :] >= h)
        forces[2, :] = forces[2, :] - fg  # * (1 - contact.float())

        # Accelerations: linear and angular accelerations computed from forces
        dvel_x = (forces / self.mass).sum(dim=1)
        dvel_omega = self.inertia_inv @ torch.cross(points - pos_x, forces).sum(dim=1)

        return dpos_x, dpos_R, dvel_x, dvel_omega, forces  # _track #torch.zeros_like(self.f)

    def forward_rigid_layer(self, t, state):
        pos_x, pos_R, vel_x, vel_omega, f_old = state

        dpos_x = vel_x
        vel_omega_skew = skew_symmetric(vel_omega).to(self.device)
        dpos_R = vel_omega_skew @ pos_R
        robot_points = pos_R @ self.robot_points + pos_x
        dpoints = vel_omega_skew @ (robot_points - pos_x) + vel_x

        # interpolate terrain properties at points of contact (POC)
        H, W = self.height.shape
        xy_grid = robot_points[0:2, :] / self.grid_res + torch.tensor([H / 2., W / 2.], device=self.device).view((2, 1))
        h_r = self.sample_by_interp(self.height, xy_grid)
        e_r = self.elasticity_rigid
        d_r = self.damping_rigid

        # contacts
        # contact_r = (robot_points[2, :] <= h_r)
        contact_r = self.soft_contact(h_r, robot_points)

        # compute normals to heightmap at POC
        idx_points_x = torch.clamp(torch.torch.floor(robot_points[0, :]).long(), min=0, max=self.height.shape[0] - 2)
        idx_points_y = torch.clamp(torch.torch.floor(robot_points[1, :]).long(), min=0, max=self.height.shape[1] - 2)
        dzx = self.height[idx_points_x + 1, idx_points_y] - self.height[idx_points_x, idx_points_y]
        dzy = self.height[idx_points_x, idx_points_y + 1] - self.height[idx_points_x, idx_points_y]
        nh = torch.cross(torch.vstack([torch.ones_like(dzx), torch.zeros_like(dzx), dzx]),
                         torch.vstack([torch.zeros_like(dzy), torch.ones_like(dzy), dzy]))
        nh = nh / nh.norm(dim=0)

        # compute terrain + gravity forces
        z = torch.tile(torch.tensor([[0], [0], [1]]), (1, robot_points.shape[1])).to(self.device)
        dist = (nh * z * (h_r - robot_points[2, :])).sum(dim=0)
        vel = (nh * dpoints).sum(dim=0)
        forces_hard = nh * (e_r * dist - d_r * vel) * contact_r

        forces = forces_hard
        fg = self.mass * self.gravity
        forces[2, :] = forces[2, :] - fg

        # Force generated by tracks at contacts
        p = robot_points.shape[1]
        vel_tracks = torch.hstack((self.vel_tracks[0] * torch.ones(1, int(robot_points.shape[1] / 2), device=self.device),
                                   self.vel_tracks[1] * torch.ones(1, int(robot_points.shape[1] / 2), device=self.device)))
        # ---- track forces in rcf
        slope_rate = 1.
        f_tx = 2 * fg * nh[2, :] * contact_r * 2 * (torch.sigmoid(slope_rate * (vel_tracks - (pos_R.t() @ dpoints)[0, :])) - 0.5)
        f_ty = 2 * fg * nh[2, :] * contact_r * 2 * (torch.sigmoid(slope_rate * (-pos_R.t() @ dpoints)[1, :]) - 0.5)
        # ---- robot pose in wcf
        rx = pos_R @ torch.vstack([torch.ones(p), torch.zeros(p), torch.zeros(p)]).to(self.device)  # unit vector in the robot x-direction in wcf
        ry = pos_R @ torch.vstack([torch.zeros(p), torch.ones(p), torch.zeros(p)]).to(self.device)  # unit vector in the robot y-direction in wcf
        # ---- robot's x,y-direction projected on terrain plane in wcf
        hx, hy = rx - (rx * nh).sum(dim=0) * nh, ry - (ry * nh).sum(dim=0) * nh
        hx, hy = hx / torch.norm(hx, dim=0), hy / torch.norm(hy, dim=0)
        # ---- track forces projected on terrain surface in wcf
        f_track = f_tx * hx + f_ty * hy
        forces = forces + f_track

        # Accelerations: linear and angular accelerations computed from forces
        dvel_x = (forces / self.mass).sum(dim=1)
        dvel_omega = self.inertia_inv @ torch.cross(robot_points - pos_x, forces).sum(dim=1)

        return dpos_x, dpos_R, dvel_x, dvel_omega, f_tx * hx  # 0.2 * (forces_soft + forces_hard)  # f_tx * hx

    def forward_rigid_soft_layers(self, t, state):
        pos_x, pos_R, vel_x, vel_omega, f_old = state

        dpos_x = vel_x
        vel_omega_skew = skew_symmetric(vel_omega).to(self.device)
        dpos_R = vel_omega_skew @ pos_R
        robot_points = pos_R @ self.robot_points + pos_x
        dpoints = vel_omega_skew @ (robot_points - pos_x) + vel_x

        # sample terrain properties at points of contact (POC)
        h, w = self.height.shape
        xy_grid = robot_points[0:2, :] / self.grid_res + torch.tensor([h / 2., w / 2.], device=self.device).view((2, 1))
        h_r = self.sample_by_interp(self.height, xy_grid)
        h = h_r + self.sample_by_interp(self.height_soft, xy_grid)
        e = self.sample_by_interp(self.elasticity, xy_grid)
        d = self.sample_by_interp(self.damping, xy_grid)
        f = self.sample_by_interp(self.friction, xy_grid)

        # contacts
        # contact = (robot_points[2, :] <= h)
        contact = self.soft_contact(h, robot_points)
        # contact_r = (robot_points[2, :] <= h_r)
        contact_r = self.soft_contact(h_r, robot_points)

        # compute normals to heightmap at POC
        idx_points_x = torch.clamp(torch.torch.floor(xy_grid[0, :]).long(), min=0, max=self.height.shape[0] - 2)
        idx_points_y = torch.clamp(torch.torch.floor(xy_grid[1, :]).long(), min=0, max=self.height.shape[1] - 2)
        dzx = self.height[idx_points_x + 1, idx_points_y] - self.height[idx_points_x, idx_points_y]
        dzy = self.height[idx_points_x, idx_points_y + 1] - self.height[idx_points_x, idx_points_y]
        nh = torch.cross(torch.vstack([torch.ones_like(dzx), torch.zeros_like(dzx), dzx]),
                         torch.vstack([torch.zeros_like(dzy), torch.ones_like(dzy), dzy]))
        nh = nh / nh.norm(dim=0)

        # compute terrain + gravity forces
        z = torch.tile(torch.tensor([[0], [0], [1]]), (1, robot_points.shape[1])).to(robot_points.device)
        dist = (nh * z * (h - robot_points[2, :])).sum(dim=0)
        vel = (nh * dpoints).sum(dim=0)
        forces_soft = nh * (e * dist - d * vel) * contact  # * (~contact_r)
        forces_hard = nh * (self.elasticity_rigid * dist - self.damping_rigid * vel) * contact_r
        forces = forces_soft + forces_hard
        fg = self.mass * self.gravity
        forces[2, :] = forces[2, :] - fg

        # Force generated by tracks at contacts
        N = robot_points.shape[1]
        vel_tracks = 1 * torch.hstack((self.vel_tracks[0] * torch.ones(1, int(robot_points.shape[1] / 2), device=self.device),
                                       self.vel_tracks[1] * torch.ones(1, int(robot_points.shape[1] / 2), device=self.device)))
        # ---- track forces in rcf
        slope_rate = 1.
        f_tx = 2 * f * fg * nh[2, :] * contact * 2 * (torch.sigmoid(slope_rate * (vel_tracks - (pos_R.t() @ dpoints)[0, :])) - 0.5)
        f_ty = 2 * f * fg * nh[2, :] * contact * 2 * (torch.sigmoid(slope_rate * (-pos_R.t() @ dpoints)[1, :]) - 0.5)
        # ---- robot pose in wcf
        rx = pos_R @ torch.vstack([torch.ones(N), torch.zeros(N), torch.zeros(N)]).to(self.device)  # unit vector in the robot x-direction in wcf
        ry = pos_R @ torch.vstack([torch.zeros(N), torch.ones(N), torch.zeros(N)]).to(self.device)  # unit vector in the robot y-direction in wcf
        # ---- robot's x,y-direction projected on terrain plane in wcf
        hx, hy = rx - (rx * nh).sum(dim=0) * nh, ry - (ry * nh).sum(dim=0) * nh
        hx, hy = hx / torch.norm(hx, dim=0), hy / torch.norm(hy, dim=0)
        # ---- track forces projected on terrain surface in wcf
        f_track = f_tx * hx + f_ty * hy
        forces = forces + f_track

        # Accelerations: linear and angular accelerations computed from forces
        dvel_x = (forces / self.mass).sum(dim=1)
        # torch.stack(((forces[0, :].sum() / self.mass).squeeze(), (forces[1, :].sum() / self.mass).squeeze(), (forces[2, :].sum() / self.mass).squeeze()))
        dvel_omega = self.inertia_inv @ torch.cross(robot_points - pos_x, forces).sum(dim=1)

        return dpos_x, dpos_R, dvel_x, dvel_omega, f_tx * hx  # forces_soft + forces_hard  # f_tx * hx

    def sample_by_interp(self, grid, coords, mode='bilinear'):
        # example:
        # im = torch.rand((4,8)).view(1,1,4,8)
        # pt = torch.tensor([[2, 2.25, 2.5, 2.75, 3,4],[1.5,1.5,1.5,1.5,1.5,1.5]], dtype=torch.double)
        H = grid.shape[0]
        W = grid.shape[1]
        WW = (W - 1) / 2
        HH = (H - 1) / 2
        coords_r = coords.clone()
        coords_r[1, ] = (coords[0, :] - HH) / HH
        coords_r[0, ] = (coords[1, :] - WW) / WW
        return torch.nn.functional.grid_sample(grid.view(1, 1, H, W),
                                               coords_r.permute(1, 0).view(1, 1, coords_r.shape[1], coords_r.shape[0]),
                                               mode=mode, align_corners=True).squeeze()

    @staticmethod
    def soft_contact(h, pos, slope_rate: float = 10.):
        z = pos[2]
        # sigmoid function
        return 1. / (1. + torch.exp(-slope_rate * (h - z)))

    def sim(self, state, tt):
        if isinstance(state, tuple):
            state = State(*state)

        pos_x, pos_R, vel_x, vel_omega, forces = state
        pos_x, pos_R, vel_x, vel_omega, forces = [pos_x], [pos_R], [vel_x], [vel_omega], [forces]
        dt = (tt[1:] - tt[:-1]).mean().item()

        for t in tt[1::]:
            dstate = self.forward(t, state)
            state = state.update(dstate, dt)

            pos_x.append(state[0])
            pos_R.append(state[1])
            vel_x.append(state[2])
            vel_omega.append(state[3])
            forces.append(state[4])
        states = [torch.stack(pos_x), torch.stack(pos_R), torch.stack(vel_x), torch.stack(vel_omega), torch.stack(forces)]

        return states

    def sim_control(self, state, goal_state, tt):
        if isinstance(state, tuple):
            state = State(*state)

        goal_xyz, rot_goal = goal_state[:2]
        goal_pose = torch.eye(4, device=self.device)
        goal_pose[:3, :3] = rot_goal
        goal_pose[:3, 3] = goal_xyz.squeeze()

        dt = (tt[1:] - tt[:-1]).mean().item()
        tracks_distance = self.robot_points[1].max() - self.robot_points[1].min()

        pos_x, pos_R, vel_x, vel_omega, forces = state
        pos_x, pos_R, vel_x, vel_omega, forces = [pos_x], [pos_R], [vel_x], [vel_omega], [forces]

        for t in tt[1:]:
            # control
            v, w = pose_control(state, goal_pose, self.Kp_rho, self.Kp_theta, self.Kp_yaw)

            # two tracks (flippers) robot model
            u1 = v + w * tracks_distance / 4.
            u2 = v - w * tracks_distance / 4.
            self.vel_tracks = torch.tensor([u1, u2])

            dstate = self.forward(t, state)
            state = state.update(dstate, dt)

            pos_x.append(state[0])
            pos_R.append(state[1])
            vel_x.append(state[2])
            vel_omega.append(state[3])
            forces.append(state[4])

        states = [torch.stack(pos_x), torch.stack(pos_R), torch.stack(vel_x), torch.stack(vel_omega),
                  torch.stack(forces)]

        return states

    def update_trajectory(self, tt=None, states=None):
        if states is None:
            assert tt is not None
            state = self.state.as_tuple()
            if self.use_ode:
                states = odeint(self.forward, state, tt, atol=1e-3, rtol=1e-3)
            else:
                states = self.sim(state, tt)

        pos_x, pos_R, vel_x, vel_omega, forces = states
        self.pos_x = pos_x.detach().cpu().numpy()
        self.pos_R = pos_R.detach().cpu().numpy()
        self.vel_x = vel_x.detach().cpu().numpy()
        self.vel_omega = vel_omega.detach().cpu().numpy()
        self.forces = forces.detach().cpu().numpy()

        # # update state
        # self.state = State(xyz=self.pos_x[-1], rot=self.pos_R[-1], vel=self.vel_x[-1], omega=self.vel_omega[-1], forces=self.forces[-1])

    def set_state(self, state):
        assert isinstance(state, State) or isinstance(state, tuple)
        if isinstance(state, State):
            self.state = state
        else:
            self.state = State(xyz=state[0], rot=state[1], vel=state[2], omega=state[3], forces=state[4])


def make_dphysics_model(height, dphys_cfg: DPhysConfig):
    system = RigidBodySoftTerrain(height=height,
                                  grid_res=dphys_cfg.grid_res,
                                  damping=dphys_cfg.damping, elasticity=dphys_cfg.elasticity, friction=dphys_cfg.friction,
                                  mass=dphys_cfg.robot_mass,
                                  state=State(xyz=dphys_cfg.robot_init_xyz),
                                  vel_tracks=dphys_cfg.vel_tracks)
    return system


def dphysics(height, controls, robot_model='husky', state=None, dphys_cfg=None, device=None):
    """
    Simulate robot-terrain interaction model
    @param height: heightmap, 2D numpy array
    @param controls: control commands, linear velocity and angular velocity
    @param robot_model: robot model, e.g., 'tradr', 'husky', 'marv'
    @param state: initial state, State
    @param dphys_cfg: configuration, DPhysConfig
    @param device: cuda or cpu
    @return: states, system
    """
    assert isinstance(height, np.ndarray)
    assert height.shape[0] == height.shape[1]
    assert isinstance(controls, dict)
    assert 'stamps' in controls.keys()
    assert 'linear_v' in controls.keys()
    assert 'angular_w' in controls.keys()
    assert isinstance(state, State) or state is None

    if dphys_cfg is None:
        dphys_cfg = DPhysConfig()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if state is None:
        state = State(xyz=torch.tensor([0., 0., 0.], device=device).view(3, 1),
                      rot=torch.eye(3, device=device),
                      vel=torch.tensor([0., 0., 0.], device=device).view(3, 1),
                      omega=torch.tensor([0., 0., 0.], device=device).view(3, 1),
                      device=device)

    """ Create robot-terrain interaction models """
    system = RigidBodySoftTerrain(height=height,
                                  grid_res=dphys_cfg.grid_res,
                                  friction=dphys_cfg.friction,
                                  mass=dphys_cfg.robot_mass,
                                  state=state,
                                  device=device, use_ode=False,
                                  interaction_model='diffdrive',
                                  robot_model=robot_model)

    # put models with their params to self.device
    system = system.to(device)
    tt = controls['stamps'].to(device)

    """ Navigation loop """
    dt = (tt[1:] - tt[:-1]).mean()

    xyz, Rs, linear_v, angular_w, forces = state
    xyz, Rs, linear_v, angular_w, forces = [xyz], [Rs], [linear_v], [angular_w], [forces]

    for t in range(len(tt[1:])):
        v, w = controls['linear_v'][t], controls['angular_w'][t]

        state[2][0] = v
        state[3][2] = w

        dstate = system.forward(t, state)
        state = state.update(dstate, dt)

        roll, pitch, yaw = rot2rpy(state[1].squeeze())
        if torch.abs(roll) > np.pi / 2. or torch.abs(pitch) > np.pi / 2.:
            warnings.warn('Robot is flipped over!')
            break

        xyz.append(state[0])
        Rs.append(state[1])
        linear_v.append(state[2])
        angular_w.append(state[3])
        forces.append(state[4])

    xyz = torch.stack(xyz)
    Rs = torch.stack(Rs)
    linear_v = torch.stack(linear_v)
    angular_w = torch.stack(angular_w)
    forces = torch.stack(forces)

    states = [xyz, Rs, linear_v, angular_w, forces]

    return states, system
