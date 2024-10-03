import torch
import numpy as np
import yaml
import os
import open3d as o3d


class DPhysConfig:
    def __init__(self, robot='tradr'):
        # robot parameters
        self.robot = robot
        if 'tradr' in robot:
            self.robot_mass = 40.  # kg
            self.vel_max = 1.0  # m/s
            self.omega_max = 0.7  # rad/s
        elif 'marv' in robot:
            self.robot_mass = 60.  # kg
            self.vel_max = 1.2  # m/s
            self.omega_max = 0.8  # rad/s
        elif 'husky' in robot:
            self.robot_mass = 50.
            self.vel_max = 1.4  # m/s
            self.omega_max = 1.0  # rad/s
        else:
            raise ValueError(f'Robot {robot} not supported. Available robots: tradr, marv, husky')
        self.robot_points, self.driving_parts, self.robot_size = self.robot_geometry(robot=robot)
        self.robot_I = self.inertia_tensor(self.robot_mass, self.robot_points)

        self.gravity = 9.81  # acceleration due to gravity, m/s^2

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
        self.integration_mode = 'rk4'  # 'euler', 'rk2', 'rk4'

    @staticmethod
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

    def get_points_from_robot_mesh(self, robot, voxel_size=0.1, return_mesh=False):
        """
        Returns the point cloud as vertices of the robot mesh.

        Parameters:
        - robot: Name of the robot.
        - voxel_size: Voxel size for downsampling the point cloud.
        - return_mesh: Whether to return the mesh object as well.

        Returns:
        - Point cloud as vertices of the robot mesh.
        """
        # TODO: add MARV mesh, using the same mesh as TRADR for now
        mesh_path = os.path.join(os.path.dirname(__file__), f'../../data/meshes/{robot}.obj')
        assert os.path.exists(mesh_path), f'Mesh file {mesh_path} does not exist.'
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        if voxel_size:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        x_points = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
        if return_mesh:
            return x_points, mesh

        return x_points

    def robot_geometry(self, robot, device='cpu'):
        """
        Returns the parameters of the rigid body.
        """
        x_points = self.get_points_from_robot_mesh(robot)
        s_x, s_y = x_points[:, 0].max() - x_points[:, 0].min(), x_points[:, 1].max() - x_points[:, 1].min()

        cog = x_points.mean(dim=0)
        if robot in ['tradr', 'tradr2']:
            # divide the point cloud into left and right parts
            mask_left = x_points[..., 1] > (cog[1] + s_y / 4.)
            mask_right = x_points[..., 1] < (cog[1] - s_y / 4.)
            # driving parts: left and right tracks
            driving_parts = [mask_left, mask_right]
        elif robot in ['marv', 'husky']:
            # divide the point cloud into front left, front right, rear left, rear right flippers / wheels
            mask_fl = torch.logical_and(x_points[..., 0] < (cog[0] - s_x / 8.),
                                        x_points[..., 1] > (cog[1] + s_y / 3.))
            mask_fr = torch.logical_and(x_points[..., 0] < (cog[0] - s_x / 8.),
                                        x_points[..., 1] < (cog[1] - s_y / 3.))
            mask_rl = torch.logical_and(x_points[..., 0] > (cog[0] + s_x / 8.),
                                        x_points[..., 1] > (cog[1] + s_y / 3.))
            mask_rr = torch.logical_and(x_points[..., 0] > (cog[0] + s_x / 8.),
                                        x_points[..., 1] < (cog[1] - s_y / 3.))
            # driving parts: front left, front right, rear left, rear right flippers / wheels
            driving_parts = [mask_fl, mask_fr, mask_rl, mask_rr]
        else:
            raise ValueError(f'Robot {robot} not supported. Available robots: tradr, marv, husky')

        # robot size
        robot_size = (s_x, s_y)

        # put tensors on the device
        x_points = x_points.to(device)
        driving_parts = [p.to(device) for p in driving_parts]

        return x_points, driving_parts, robot_size

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


def show_robot():
    import matplotlib
    import open3d as o3d
    matplotlib.use('Qt5Agg')

    robot = 'marv'
    cfg = DPhysConfig(robot=robot)
    points = cfg.robot_points
    points_driving = [points[mask] for mask in cfg.driving_parts]

    pcd_driving = o3d.geometry.PointCloud()
    pcd_driving.points = o3d.utility.Vector3dVector(torch.vstack(points_driving))
    pcd_driving.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([pcd_driving])

    mesh = cfg.get_points_from_robot_mesh(robot, return_mesh=True)[1]

    o3d.visualization.draw_geometries([mesh, pcd_driving])


def save_cfg():
    cfg = DPhysConfig()
    cfg.to_yaml('../../config/dphys_cfg.yaml')


if __name__ == '__main__':
    show_robot()
    # save_cfg()
