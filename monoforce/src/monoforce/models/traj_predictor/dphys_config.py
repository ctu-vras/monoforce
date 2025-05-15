import torch
import numpy as np
import yaml
import os
import open3d as o3d


def get_points_from_robot_mesh(robot, voxel_size=0.1, return_mesh=False):
    """
    Returns the point cloud as vertices of the robot mesh.

    Parameters:
    - robot: Name of the robot.
    - voxel_size: Voxel size for downsampling the point cloud.
    - return_mesh: Whether to return the mesh object as well.

    Returns:
    - Point cloud as vertices of the robot mesh.
    """
    if 'tradr' in robot:
        robot = 'tradr'
    elif 'marv' in robot:
        robot = 'marv'
    mesh_path = os.path.join(os.path.dirname(__file__), f'../../../../config/meshes/{robot}.obj')
    assert os.path.exists(mesh_path), f'Mesh file {mesh_path} does not exist.'
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    x_points = torch.as_tensor(np.asarray(pcd.points), dtype=torch.float32)
    if return_mesh:
        return x_points, mesh

    return x_points


def robot_geometry(robot):
    """
    Returns the parameters of the rigid body.
    """
    x_points = get_points_from_robot_mesh(robot)
    s_x, s_y = x_points[:, 0].max() - x_points[:, 0].min(), x_points[:, 1].max() - x_points[:, 1].min()

    cog = x_points.mean(dim=0)
    if robot in ['tradr', 'tradr2']:
        # divide the point cloud into left and right parts
        mask_l = (x_points[..., 1] > (cog[1] + s_y / 4.)) & (x_points[..., 2] < cog[2])
        mask_r = (x_points[..., 1] < (cog[1] - s_y / 4.)) & (x_points[..., 2] < cog[2])
        # driving parts: left and right tracks
        driving_parts = [mask_l, mask_r]
    elif robot in ['marv', 'husky', 'husky_oru']:
        # divide the point cloud into front left, front right, rear left, rear right flippers / wheels
        mask_fl = (x_points[..., 0] > (cog[0] + s_x / 8.)) & \
                  (x_points[..., 1] > (cog[1] + s_y / 3.))
        mask_fr = (x_points[..., 0] > (cog[0] + s_x / 8.)) & \
                  (x_points[..., 1] < (cog[1] - s_y / 3.))
        mask_rl = (x_points[..., 0] < (cog[0] - s_x / 8.)) & \
                  (x_points[..., 1] > (cog[1] + s_y / 3.))
        mask_rr = (x_points[..., 0] < (cog[0] - s_x / 8.)) & \
                  (x_points[..., 1] < (cog[1] - s_y / 3.))
        # driving parts: front left, front right, rear left, rear right flippers / wheels
        driving_parts = [mask_fl, mask_fr, mask_rl, mask_rr]
    else:
        raise ValueError(f'Robot {robot} not supported. Available robots: tradr, marv, husky')

    # robot size
    robot_size = (s_x, s_y)

    # put tensors on the device
    x_points = x_points
    driving_parts = [p for p in driving_parts]

    return x_points, driving_parts, robot_size


class DPhysConfig:
    def __init__(self, robot='marv', grid_res=0.1):
        # robot parameters
        self.robot = robot
        self.vel_max = 1.0  # m/s
        self.omega_max = 2.0  # rad/s
        if 'tradr' in robot:
            self.robot_mass = 40.  # kg
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
        elif 'husky' in robot:
            self.robot_mass = 50.
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
        self.robot_points, self.driving_parts, self.robot_size = robot_geometry(robot=robot)

        self.gravity = 9.81  # acceleration due to gravity, m/s^2
        self.gravity_direction = torch.tensor([0., 0., -1.])  # gravity direction in the world frame

        # height map parameters
        self.grid_res = grid_res  # grid resolution of the heightmap, [m]
        self.r_min = 0.6  # minimum distance of the terrain from the robot, [m]
        self.d_max = 6.4  # half-size of the terrain, heightmap range: [-d_max, d_max]
        self.h_max = 2.0  # maximum height of the terrain, heightmap range: [-h_max, h_max]
        x_grid = torch.arange(-self.d_max, self.d_max, self.grid_res)
        y_grid = torch.arange(-self.d_max, self.d_max, self.grid_res)
        self.x_grid, self.y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')
        self.z_grid = torch.zeros_like(self.x_grid)
        self.friction = 1.0 * torch.ones_like(self.z_grid)  # friction of the terrain
        self.stiffness = 50_000.  # stiffness of the terrain, [N/m]
        self.damping = np.sqrt(4 * self.robot_mass * self.stiffness)  # critical damping
        self.hm_interp_method = None

        # trajectory shooting parameters
        self.traj_sim_time = 5.0
        self.dt = 0.01
        self.n_sim_trajs = 64
        self.integration_mode = 'euler'  # 'euler', 'rk4'

        # using odeint for integration or not, from torchdiffeq: https://github.com/rtqichen/torchdiffeq
        self.use_odeint = True

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
    import open3d as o3d

    robot = 'marv'
    dphys_cfg = DPhysConfig(robot=robot)
    points = dphys_cfg.robot_points.cpu()
    points_driving = [points[mask.cpu()] for mask in dphys_cfg.driving_parts]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.0, 0.0, 1.0])

    pcd_driving = o3d.geometry.PointCloud()
    pcd_driving.points = o3d.utility.Vector3dVector(torch.vstack(points_driving))
    pcd_driving.paint_uniform_color([1.0, 0.0, 0.0])

    mesh = get_points_from_robot_mesh(robot, return_mesh=True)[1]

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
    # o3d.visualization.draw_geometries([mesh, pcd_driving, base_link_sphere] + joint_poses)
    # o3d.visualization.draw_geometries([pcd, pcd_driving] + joint_poses)
    o3d.visualization.draw_geometries([pcd, pcd_driving])
    # o3d.visualization.draw_geometries([mesh, pcd, pcd_driving])


def save_cfg():
    cfg = DPhysConfig()
    cfg.to_yaml('../../../../config/dphys_cfg.yaml')


if __name__ == '__main__':
    show_robot()
    # save_cfg()
