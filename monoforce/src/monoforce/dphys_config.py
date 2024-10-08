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
            self.joint_positions = {
                'fl': [0.250, 0.272, 0.019],
                'fr': [0.250, -0.272, 0.019],
                'rl': [-0.250, 0.272, 0.019],
                'rr': [-0.250, -0.272, 0.019]
            }
            self.joint_angles = {
                'fl': 0.0,
                'fr': 0.0,
                'rl': 0.0,
                'rr': 0.0
            }
        elif 'marv' in robot:
            self.robot_mass = 60.  # kg
            self.vel_max = 1.2  # m/s
            self.omega_max = 0.8  # rad/s
            self.joint_positions = {
                'fl': [0.250, 0.272, 0.019],
                'fr': [0.250, -0.272, 0.019],
                'rl': [-0.250, 0.272, 0.019],
                'rr': [-0.250, -0.272, 0.019]
            }
            self.joint_angles = {
                'fl': -1.0,
                'fr': -1.0,
                'rl': 1.0,
                'rr': 1.0
            }
        elif 'husky' in robot:
            self.robot_mass = 50.
            self.vel_max = 1.4  # m/s
            self.omega_max = 1.0  # rad/s
            self.joint_positions = {
                'fl': [0.256, 0.285, 0.033],
                'fr': [0.256, -0.285, 0.033],
                'rl': [-0.256, 0.285, 0.033],
                'rr': [-0.256, -0.285, 0.033]
            }
            self.joint_angles = {
                'fl': 0.0,
                'fr': 0.0,
                'rl': 0.0,
                'rr': 0.0
            }
        else:
            raise ValueError(f'Robot {robot} not supported. Available robots: tradr, marv, husky')
        self.robot_points, self.driving_parts, self.robot_size = self.robot_geometry(robot=robot)
        if 'marv' in robot:
            self.update_driving_parts()
        self.robot_I = self.inertia_tensor(self.robot_mass, self.robot_points)

        self.gravity = 9.81  # acceleration due to gravity, m/s^2

        # height map parameters
        self.grid_res = 0.1
        self.d_min = 1.0
        self.d_max = 6.4
        self.h_max_above_ground = 1.0  # above ground frame (base_footprint)
        self.k_stiffness = 10_000.
        self.k_damping = float(np.sqrt(4 * self.robot_mass * self.k_stiffness))  # critical damping
        self.k_friction = 1.0
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
        mesh_path = os.path.join(os.path.dirname(__file__), f'../../config/meshes/{robot}.obj')
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

    def get_points(self, s_x=0.5, s_y=0.5):
        x_points = torch.stack([
            torch.hstack([torch.linspace(-s_x / 2., s_x / 2., 16 // 2),
                          torch.linspace(-s_x / 2., s_x / 2., 16 // 2)]),
            torch.hstack([s_y / 2. * torch.ones(16 // 2),
                          -s_y / 2. * torch.ones(16 // 2)]),
            torch.hstack([torch.tensor([0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2]),
                          torch.tensor([0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2])])
        ]).T
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
            mask_l = x_points[..., 1] > (cog[1] + s_y / 4.)
            mask_r = x_points[..., 1] < (cog[1] - s_y / 4.)
            # driving parts: left and right tracks
            driving_parts = [mask_l, mask_r]
        elif robot in ['marv', 'husky']:
            # divide the point cloud into front left, front right, rear left, rear right flippers / wheels
            mask_fl = torch.logical_and(x_points[..., 0] > (cog[0] + s_x / 8.),
                                        x_points[..., 1] > (cog[1] + s_y / 3.))
            mask_fr = torch.logical_and(x_points[..., 0] > (cog[0] + s_x / 8.),
                                        x_points[..., 1] < (cog[1] - s_y / 3.))
            mask_rl = torch.logical_and(x_points[..., 0] < (cog[0] - s_x / 8.),
                                        x_points[..., 1] > (cog[1] + s_y / 3.))
            mask_rr = torch.logical_and(x_points[..., 0] < (cog[0] - s_x / 8.),
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

    def update_driving_parts(self):
        """
        Update the driving parts according to the joint angles
        """
        assert self.robot in ['marv'], 'Only MARV is supported for now.'
        # rotate driving parts according to joint angles
        for i, (angle, xyz) in enumerate(zip(self.joint_angles.values(), self.joint_positions.values())):
            # rotate around y-axis of the joint position
            xyz = torch.tensor(xyz)
            R = torch.tensor([[np.cos(angle), 0, np.sin(angle)],
                              [0, 1, 0],
                              [-np.sin(angle), 0, np.cos(angle)]]).float()
            mask = self.driving_parts[i]
            points = self.robot_points[mask]
            points -= xyz
            points = points @ R.T
            points += xyz
            self.robot_points[mask] = points

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
    dphys_cfg = DPhysConfig(robot=robot)
    points = dphys_cfg.robot_points
    points_driving = [points[mask] for mask in dphys_cfg.driving_parts]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.0, 0.0, 1.0])

    pcd_driving = o3d.geometry.PointCloud()
    pcd_driving.points = o3d.utility.Vector3dVector(torch.vstack(points_driving))
    pcd_driving.paint_uniform_color([1.0, 0.0, 0.0])

    mesh = dphys_cfg.get_points_from_robot_mesh(robot, return_mesh=True)[1]

    joint_poses = []
    for joint in dphys_cfg.joint_positions.values():
        # sphere
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(joint)
        sphere.paint_uniform_color([0.0, 1.0, 0.0])
        joint_poses.append(sphere)
    base_link_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    base_link_sphere.translate([0, 0, 0])
    base_link_sphere.paint_uniform_color([0.0, 0.0, 1.0])

    # visualize
    o3d.visualization.draw_geometries([mesh, pcd_driving, base_link_sphere] + joint_poses)
    # o3d.visualization.draw_geometries([pcd, pcd_driving] + joint_poses)
    # o3d.visualization.draw_geometries([pcd, pcd_driving])
    # o3d.visualization.draw_geometries([mesh, pcd, pcd_driving])


def save_cfg():
    cfg = DPhysConfig()
    cfg.to_yaml('../../config/dphys_cfg.yaml')


if __name__ == '__main__':
    show_robot()
    # save_cfg()
