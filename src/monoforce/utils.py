import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
from PIL import Image, ImageFile
import rospy
from timeit import default_timer as timer
import torch
import torchvision
import yaml
import open3d as o3d
ImageFile.LOAD_TRUNCATED_IMAGES = True


__all__ = [
    'slots',
    'timing',
    'read_points_ply',
    'read_points_bin',
    'read_points_labels',
    'normalize',
    'skew_symmetric',
    'read_yaml',
    'write_to_yaml',
    'str2bool',
]

color_palette = {
    0: {"color": [0, 0, 0], "name": "void"},
    1: {"color": [108, 64, 20], "name": "dirt"},
    3: {"color": [0, 102, 0], "name": "grass"},
    4: {"color": [0, 255, 0], "name": "tree"},
    5: {"color": [0, 153, 153], "name": "pole"},
    6: {"color": [0, 128, 255], "name": "water"},
    7: {"color": [0, 0, 255], "name": "sky"},
    8: {"color": [255, 255, 0], "name": "vehicle"},
    9: {"color": [255, 0, 127], "name": "object"},
    10: {"color": [64, 64, 64], "name": "asphalt"},
    12: {"color": [255, 0, 0], "name": "building"},
    15: {"color": [102, 0, 0], "name": "log"},
    17: {"color": [204, 153, 255], "name": "person"},
    18: {"color": [102, 0, 204], "name": "fence"},
    19: {"color": [255, 153, 204], "name": "bush"},
    23: {"color": [170, 170, 170], "name": "concrete"},
    27: {"color": [41, 121, 255], "name": "barrier"},
    31: {"color": [134, 255, 239], "name": "puddle"},
    33: {"color": [99, 66, 34], "name": "mud"},
    34: {"color": [110, 22, 138], "name": "rubble"}
}


def slots(msg):
    """Return message attributes (slots) as list."""
    return [getattr(msg, var) for var in msg.__slots__]


def timing(f):
    def timing_wrapper(*args, **kwargs):
        t0 = timer()
        ret = f(*args, **kwargs)
        t1 = timer()
        rospy.logdebug('%s %.6f s' % (f.__name__, t1 - t0))
        return ret
    return timing_wrapper


def read_points_ply(path, dtype=np.float32):
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    assert points.shape[1] == 3
    points = unstructured_to_structured(points.astype(dtype=dtype), names=['x', 'y', 'z'])
    del pcd
    return points


def read_points_bin(path, dtype=np.float32):
    xyzi = np.fromfile(path, dtype=dtype)
    xyzi = xyzi.reshape((-1, 4))
    points = unstructured_to_structured(xyzi.astype(dtype=dtype), names=['x', 'y', 'z', 'i'])
    return points


def read_points_labels(path, dtype=np.uint32):
    label = np.fromfile(path, dtype=dtype)
    label = label.reshape((-1, 1))
    # label = convert_label(label, inverse=False)
    label = unstructured_to_structured(label.astype(dtype=dtype), names=['label'])
    return label


def read_points(path, dtype=np.float32):
    # https://stackoverflow.com/questions/5899497/how-can-i-check-the-extension-of-a-file
    if path.lower().endswith('.ply'):
        points = read_points_ply(path, dtype)
    elif path.lower().endswith('.bin'):
        points = read_points_bin(path, dtype)
    else:
        raise ValueError('Cloud file must have .ply or .bin extension')
    return points


def read_poses(path, zero_origin=True):
    data = np.genfromtxt(path)
    poses = np.asarray([np.eye(4) for _ in range(len(data))]).reshape([-1, 4, 4])
    poses[:, :3, :4] = data.reshape([-1, 3, 4])
    del data
    # transform to 0-origin (pose[0] = np.eye(4))
    if zero_origin:
        poses = np.einsum("ij,njk->nik", np.linalg.inv(poses[0]), poses)
    return poses


def read_rgb(path):
    img = Image.open(path)
    img = np.asarray(img, dtype=np.uint8)
    return img


def read_intrinsics(path):
    data = np.loadtxt(path)
    K = np.zeros((3, 3))
    K[0, 0] = data[0]
    K[1, 1] = data[1]
    K[2, 2] = 1
    K[0, 2] = data[2]
    K[1, 2] = data[3]
    return K


def read_extrinsics(path, key='os1_cloud_node-pylon_camera_node'):
    """
    Transformation between camera and lidar
    """
    from scipy.spatial.transform import Rotation
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    q = data[key]['q']
    q = np.array([q['x'], q['y'], q['z'], q['w']])
    t = data[key]['t']
    t = np.array([t['x'], t['y'], t['z']])
    R_vc = Rotation.from_quat(q)
    R_vc = R_vc.as_matrix()

    RT = np.eye(4, 4)
    RT[:3, :3] = R_vc
    RT[:3, -1] = t
    RT = np.linalg.inv(RT)
    return RT


def depth_color(val, min_d=0, max_d=120):
    np.clip(val, 0, max_d, out=val)
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)


def filter_camera_points(points, img_width, img_height, K, RT, give_mask=False):
    assert points.shape[1] == 3
    RT = np.asarray(RT)
    fov_x = 2 * np.arctan2(img_width, 2 * K[0, 0]) * 180 / np.pi + 10
    fov_y = 2 * np.arctan2(img_height, 2 * K[1, 1]) * 180 / np.pi + 10
    p_l = np.ones((points.shape[0], points.shape[1] + 1))
    p_l[:, :3] = points
    p_c = np.matmul(RT, p_l.T)
    p_c = p_c.T
    x = p_c[:, 0]
    y = p_c[:, 1]
    z = p_c[:, 2]
    xangle = np.arctan2(x, z) * 180 / np.pi
    yangle = np.arctan2(y, z) * 180 / np.pi
    mask_x = (xangle > -fov_x / 2) & (xangle < fov_x / 2)
    mask_y = (yangle > -fov_y / 2) & (yangle < fov_y / 2)
    mask = mask_x & mask_y
    points_res = p_l[mask, :3]
    points_res = np.array(points_res)
    x = points_res[:, 0]
    y = points_res[:, 1]
    z = points_res[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    color = depth_color(dist, 0, 10)
    if give_mask:
        return points_res, color, mask
    return points_res, color


def read_semseg(path, label_size=None):
    def convert_label(label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in color_palette.items():
                label[temp == k["color"]] = v
        else:
            label = np.zeros(temp.shape + (3,))
            for k, v in color_palette.items():
                label[temp == k, :] = v["color"]
        return label

    semseg = Image.open(path)
    if label_size is not None:
        if label_size[0] != semseg.size[0] or label_size[1] != semseg.size[1]:
            semseg = semseg.resize((label_size[1], label_size[0]), Image.NEAREST)
            semseg = np.array(semseg)[:, :, 0]
    semseg = np.array(semseg, dtype=np.uint8)
    semseg = convert_label(semseg, False)
    return np.array(semseg, dtype=np.uint8)


def normalize(x, qlow=0., qhigh=1., eps=1e-6):
    """Scale to range 0..1"""
    if isinstance(x, torch.Tensor):
        x_max = torch.quantile(x, qhigh).item()
        x_min = torch.quantile(x, qlow).item()
        x = (x - x_min) / np.max([(x_max - x_min), eps])
        x = x.clamp(0, 1)
    else:
        x_max = np.percentile(x, 100 * qhigh)
        x_min = np.percentile(x, 100 * qlow)
        x = (x - x_min) / np.max([(x_max - x_min), eps])
        x = x.clip(0, 1)
    return x


def create_model(architecture, n_inputs, n_outputs, pretrained_backbone=True):
    assert architecture in ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101',
                            'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large']

    print('Creating model %s with %i inputs and %i outputs' % (architecture, n_inputs, n_outputs))
    Architecture = eval('torchvision.models.segmentation.%s' % architecture)
    model = Architecture(pretrained=pretrained_backbone)

    arch = architecture.split('_')[0]
    encoder = '_'.join(architecture.split('_')[1:])

    # Change input layer to accept n_inputs
    if encoder == 'mobilenet_v3_large':
        model.backbone['0'][0] = torch.nn.Conv2d(n_inputs, 16,
                                                 kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    else:
        model.backbone['conv1'] = torch.nn.Conv2d(n_inputs, 64,
                                                  kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Change final layer to output n classes
    if arch == 'lraspp':
        model.classifier.low_classifier = torch.nn.Conv2d(40, n_outputs, kernel_size=(1, 1), stride=(1, 1))
        model.classifier.high_classifier = torch.nn.Conv2d(128, n_outputs, kernel_size=(1, 1), stride=(1, 1))
    elif arch == 'fcn':
        model.classifier[-1] = torch.nn.Conv2d(512, n_outputs, kernel_size=(1, 1), stride=(1, 1))
    elif arch == 'deeplabv3':
        model.classifier[-1] = torch.nn.Conv2d(256, n_outputs, kernel_size=(1, 1), stride=(1, 1))

    return model


def skew_symmetric(x):
    U = torch.as_tensor([[0., -x[2], x[1]],
                         [x[2], 0., -x[0]],
                         [-x[1], x[0], 0.]], device=x.device)
    return U


def read_yaml(path):
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


def write_to_yaml(cfg: dict, path):
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)


def str2bool(v):
    return v.lower() in ('1', 'yes', 'true', 't', 'y')
