import rospy
import torch
import numpy as np
import yaml


class Config:
    def __init__(self):
        # height map parameters
        self.grid_res = 0.1
        self.d_min = 0.6
        self.d_max = 6.4
        self.h_above_lidar = 0.3
        self.damping = 0.
        self.elasticity = 0.
        self.friction = 0.9
        self.hm_interp_method = 'nearest'

        # robot parameters
        self.vel_tracks = [0., 0.]
        self.robot_mass = 10.
        self.num_robot_points = 5
        self.robot_inertia = (5. * np.eye(3)).tolist()
        self.robot_init_xyz = [0., 0., 1.]
        self.robot_init_q = [0., 0., 0., 1.]

        # training parameters
        self.use_terrain_cnn = False
        self.device = torch.device('cpu')
        self.lr = 0.001
        self.weight_decay = 0.0
        self.total_sim_time = 10.0
        self.n_samples = 100 * int(self.total_sim_time)
        self.sample_len = 10
        self.n_train_iters = 100
        self.convergence_std = 1e-3
        self.convergence_n_samples = 10
        self.dataset_path = '/tmp'
        self.trans_cost_weight = 1.
        self.rot_cost_weight = 1.

        # control parameters
        self.robot_terrain_interaction_model = 'diffdrive'
        self.max_vel = 2.  # m/s
        self.max_omega = 2.  # rad/s

        # MonoDem parameters
        self.img_size = (512, 512)

    def __str__(self):
        return str(self.__dict__)

    def to_rosparam(self):
        # make class attributes available as rosparams
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                v = v.tolist()
            rospy.set_param('~' + k, v)

    def from_rosparams(self, node_name):
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
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def from_yaml(self, path):
        with open(path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        for k, v in params.items():
            setattr(self, k, v)
