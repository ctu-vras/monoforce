import torch
import numpy as np
import yaml


def inertia_tensor(mass, points):
    """
    Compute the inertia tensor for a rigid body represented by point masses.

    Parameters:
    mass (float): The total mass of the body.
    points (array-like): A list or array of points (x, y, z) representing the mass distribution.
                         Each point contributes equally to the total mass.

    Returns:
    torch.Tensor: A 3x3 inertia tensor matrix.
    """

    # Convert points to a tensor
    points = torch.as_tensor(points)

    # Number of points
    n_points = points.shape[0]

    # Mass per point: assume uniform mass distribution
    mass_per_point = mass / n_points

    # Initialize the inertia tensor components
    Ixx = Iyy = Izz = Ixy = Ixz = Iyz = 0.0

    # Loop over each point and accumulate the inertia tensor components
    for x, y, z in points:
        Ixx += mass_per_point * (y ** 2 + z ** 2)
        Iyy += mass_per_point * (x ** 2 + z ** 2)
        Izz += mass_per_point * (x ** 2 + y ** 2)
        Ixy -= mass_per_point * x * y
        Ixz -= mass_per_point * x * z
        Iyz -= mass_per_point * y * z

    # Construct the inertia tensor matrix
    I = torch.tensor([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ])

    return I


class DPhysConfig:
    def __init__(self):
        # robot parameters
        self.robot_mass = 40.  # kg
        self.robot_size = (1.0, 0.5)  # length, width in meters
        self.robot_points, self.robot_mask_left, self.robot_mask_right = self.rigid_body_geometry(from_mesh=False)
        self.robot_I = inertia_tensor(self.robot_mass, self.robot_points)  # 3x3 inertia tensor, kg*m^2
        self.robot_I *= 10.  # increase inertia for stability, as the point cloud is very sparse
        self.vel_max = 1.2  # m/s
        self.omega_max = 0.4  # rad/s

        # height map parameters
        self.grid_res = 0.1
        self.d_min = 1.0
        self.d_max = 6.4
        self.h_max_above_ground = 1.0  # above ground frame (base_footprint)
        self.k_stiffness = 5_000.
        self.k_damping = float(np.sqrt(4 * self.robot_mass * self.k_stiffness))  # critical damping
        self.k_friction = 0.5
        self.hm_interp_method = None

        # trajectory shooting parameters
        self.traj_sim_time = 5.0
        self.dt = 0.01
        self.n_sim_trajs = 32
        self.integration_mode = 'euler'  # 'euler', 'rk2', 'rk4'

    def rigid_body_geometry(self, from_mesh=False):
        """
        Returns the parameters of the rigid body.
        """
        if from_mesh:
            import open3d as o3d
            robot = 'tradr'
            mesh_file = f'../../data/meshes/{robot}.obj'
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            n_points = 128
            x_points = np.asarray(mesh.sample_points_uniformly(n_points).points)
            x_points = torch.tensor(x_points)
        else:
            size = self.robot_size
            s_x, s_y = size
            x_points = torch.stack([
                torch.hstack([torch.linspace(-s_x / 2., s_x / 2., 16 // 2),
                              torch.linspace(-s_x / 2., s_x / 2., 16 // 2)]),
                torch.hstack([s_y / 2. * torch.ones(16 // 2),
                              -s_y / 2. * torch.ones(16 // 2)]),
                torch.hstack([torch.tensor([0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2]),
                              torch.tensor([0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2])])
            ]).T

        # divide the point cloud into left and right parts
        cog = x_points.mean(dim=0)
        mask_left = x_points[..., 1] > cog[1]
        mask_right = x_points[..., 1] < cog[1]

        return x_points, mask_left, mask_right

    def __str__(self):
        return str(self.__dict__)

    def to_rosparam(self):
        import rospy
        # make class attributes available as rosparams
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            rospy.set_param('~' + k, v)

    def from_rosparams(self, node_name):
        import rospy
        # make rosparams available as class attributes
        for k in rospy.get_param_names():
            if k.startswith('/' + node_name):
                setattr(self, k.split('/')[-1], rospy.get_param(k))

    def to_yaml(self, path):
        # go through all class attributes
        for k, v in self.__dict__.items():
            # if they are np arrays or torch tensors, convert them to lists
            if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                setattr(self, k, v.tolist())

        with open(path, 'w') as f:
            yaml.safe_dump(self.__dict__, f)

    def from_yaml(self, path):
        with open(path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        for k, v in params.items():
            setattr(self, k, v)


if __name__ == '__main__':
    cfg = DPhysConfig()
    cfg.to_yaml('../../config/dphys_cfg.yaml')
