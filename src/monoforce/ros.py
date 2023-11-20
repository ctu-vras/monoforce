import torch
import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm
from std_msgs.msg import Header
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from tf.transformations import quaternion_from_matrix, euler_from_quaternion, quaternion_matrix
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from grid_map_msgs.msg import GridMap
from sensor_msgs.msg import PointCloud2
from ros_numpy import msgify
from numpy.lib.recfunctions import unstructured_to_structured
from tf2_ros import BufferCore
from rosbag import Bag, ROSBagException
import rospy


def xyzR_to_pose_msg(xyz, R):
    T = np.eye(4)
    T[:3, :3] = R
    xyz = xyz.squeeze()
    msg = Pose(Point(*xyz), Quaternion(*quaternion_from_matrix(T)))
    return msg

def traj_to_path_msg(pos_x, pos_R):
    # assert isinstance(pos_x, torch.Tensor) and isinstance(pos_R, torch.Tensor)
    N = len(pos_x)
    assert pos_x.shape == (N, 3, 1)
    assert pos_R.shape == (N, 3, 3)
    if isinstance(pos_x, torch.Tensor):
        pos_x = pos_x.detach().cpu().numpy()
    if isinstance(pos_R, torch.Tensor):
        pos_R = pos_R.detach().cpu().numpy()
    msg = Path()
    msg.poses = [PoseStamped(Header(), xyzR_to_pose_msg(xyz, R)) for xyz, R in zip(pos_x, pos_R)]
    return msg

def height_map_to_gridmap_msg(heightmap, grid_res,
                              xyz=np.array([0, 0, 0]), q=np.array([0., 0., 0., 1.])):
    assert isinstance(heightmap, np.ndarray)
    assert heightmap.ndim == 2
    map = GridMap()

    H, W = heightmap.shape

    # TODO: remove this hack
    # array is visualized flipped
    heightmap = np.flip(heightmap, axis=1)
    # somehow rotation does not get applied, do it manually here:
    angle = euler_from_quaternion(q)[2]
    heightmap = rotate(heightmap, np.rad2deg(angle) - 90, reshape=False)

    multi_array = Float32MultiArray()
    multi_array.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
    multi_array.layout.data_offset = 0

    multi_array.layout.dim[0].label = "column_index"
    multi_array.layout.dim[0].size = W
    multi_array.layout.dim[0].stride = W * H

    multi_array.layout.dim[1].label = "row_index"
    multi_array.layout.dim[1].size = H
    multi_array.layout.dim[1].stride = H

    multi_array.data = heightmap.flatten().tolist()
    
    map.layers.append("elevation")
    map.data.append(multi_array)
    map.info.length_x = H * grid_res
    map.info.length_y = W * grid_res
    map.info.pose = Pose(Point(*xyz), Quaternion(*q))
    map.info.resolution = grid_res
    return map

def height_map_to_point_cloud_msg(height, grid_res, xyz=np.asarray([0., 0., 0.]), q=np.asarray([0., 0., 0., 1.])):
    assert isinstance(height, np.ndarray)
    assert height.ndim == 2
    h, w = height.shape
    n_pts = h * w
    y, x = np.meshgrid(np.arange(-h//2, h//2), np.arange(-w//2, w//2))
    x = x.ravel() * grid_res
    y = y.ravel() * grid_res
    z = height.ravel()
    pts = np.concatenate([x[None], y[None], z[None]], axis=0).T
    # transform points using xyz and q
    if not np.allclose(xyz, 0) or not np.allclose(q, np.array([0., 0., 0., 1.])):
        T = np.eye(4)
        T[:3, :3] = quaternion_matrix(q)[:3, :3]
        T[:3, 3] = xyz
        pts = pts @ T[:3, :3].T + T[:3, 3]
    assert pts.shape == (n_pts, 3)
    pts = np.asarray(pts, dtype='float32')
    cloud = unstructured_to_structured(pts, names=['x', 'y', 'z'])
    msg = msgify(PointCloud2, cloud)
    return msg

def load_tf_buffer(bag_paths, tf_topics=None):
    if tf_topics is None:
        tf_topics = ['/tf', '/tf_static']

    # tf_buffer = BufferCore(cache_time=rospy.Duration(2**31 - 1))
    # tf_buffer = BufferCore(cache_time=rospy.Duration(24 * 60 * 60))
    tf_buffer = BufferCore(rospy.Duration(24 * 60 * 60))

    for path in bag_paths:
        try:
            with Bag(path, 'r') as bag:
                for topic, msg, stamp in tqdm(bag.read_messages(topics=tf_topics),
                                              desc='%s: reading transforms' % path.split('/')[-1],
                                              total=bag.get_message_count(topic_filters=tf_topics)):
                    if topic == '/tf':
                        for tf in msg.transforms:
                            tf_buffer.set_transform(tf, 'bag')
                    elif topic == '/tf_static':
                        for tf in msg.transforms:
                            tf_buffer.set_transform_static(tf, 'bag')

        except ROSBagException as ex:
            print('Could not read %s: %s' % (path, ex))

    return tf_buffer
