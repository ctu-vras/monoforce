import torch
import numpy as np
import yaml


class DPhysConfig:
    def __init__(self):
        # height map parameters
        self.grid_res = 0.1
        self.d_min = 1.0
        self.d_max = 6.4
        self.h_max_above_ground = 1.0  # above ground frame (base_footprint)
        self.damping = 0.
        self.elasticity = 0.
        self.friction = 0.9
        self.hm_interp_method = None

        # robot parameters
        self.vel_tracks = [0., 0.]
        self.robot_mass = 10.
        self.robot_inertia = (5. * np.eye(3)).tolist()
        self.robot_init_xyz = [0., 0., 1.]
        self.robot_init_q = [0., 0., 0., 1.]
        self.robot_size = (1.0, 0.6)

        # training parameters
        self.total_sim_time = 10.0
        self.n_samples = 100 * int(self.total_sim_time)
        self.sample_len = 10

        # control parameters
        self.robot_terrain_interaction_model = 'diffdrive'
        self.max_vel = 2.  # m/s
        self.max_omega = 2.  # rad/s

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
