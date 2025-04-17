from __future__ import division, absolute_import, print_function
import rclpy
import torch
import numpy as np
from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge
from monoforce.utils import slots
from geometry_msgs.msg import Pose, Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from grid_map_msgs.msg import GridMap
from sensor_msgs.msg import CompressedImage, Image
from visualization_msgs.msg import Marker


def numpy_to_gridmap_layer(data: np.ndarray):
    assert data.ndim == 2
    data_array = Float32MultiArray()
    data_array.layout.dim.append(MultiArrayDimension())
    data_array.layout.dim.append(MultiArrayDimension())
    data_array.layout.dim[0].label = 'column_index'
    data_array.layout.dim[0].size = data.shape[0]
    data_array.layout.dim[0].stride = data.shape[0] * data.shape[1]
    data_array.layout.dim[1].label = 'row_index'
    data_array.layout.dim[1].size = data.shape[1]
    data_array.layout.dim[1].stride = data.shape[1]
    data_array.data = rotate(data, 180).flatten().tolist()
    return data_array

def terrain_to_gridmap_msg(layers: list[np.ndarray], layer_names: list[str],
                           grid_res: float,
                           xyz=None, q=None):
    if q is None:
        q = [0., 0., 0., 1.]
    if xyz is None:
        xyz = [0., 0., 0.]
    map = GridMap()
    for layer, layer_name in zip(layers, layer_names):
        assert layer.ndim == 2
        map.layers.append(layer_name)
        map.data.append(numpy_to_gridmap_layer(layer))

    H, W = layers[0].shape
    map.info.resolution = grid_res
    map.info.length_x = W * grid_res
    map.info.length_y = H * grid_res
    map.info.pose.position.x = float(xyz[0])
    map.info.pose.position.y = float(xyz[1])
    map.info.pose.position.z = float(xyz[2])
    map.info.pose.orientation.x = float(q[0])
    map.info.pose.orientation.y = float(q[1])
    map.info.pose.orientation.z = float(q[2])
    map.info.pose.orientation.w = float(q[3])

    return map


def poses_to_marker(poses, color=None):
    assert isinstance(poses, np.ndarray) or isinstance(poses, torch.Tensor)
    assert poses.ndim == 3 or poses.ndim == 2
    if poses.ndim == 3:
        assert poses.shape[1:] == (4, 4)
    elif poses.ndim == 2:
        assert poses.shape[1] == 7
    marker = Marker()
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.03
    if color is not None:
        assert len(color) == 3
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
    marker.color.a = 1.0
    for t in range(poses.shape[0]):
        p = poses[t]
        if p.shape == (4, 4):
            pose_msg = Pose()
            pose_msg.position.x = float(p[0, 3])
            pose_msg.position.y = float(p[1, 3])
            pose_msg.position.z = float(p[2, 3])
            marker.points.append(pose_msg.position)
        elif p.shape == (7,):
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = float(p[2])
            marker.points.append(pose.position)
    return marker


def rgb_msg_to_cv2(msg, cv_bridge=CvBridge()):
    img_msg = CompressedImage(*slots(msg))
    # convert compressed image message to numpy array
    img = cv_bridge.compressed_imgmsg_to_cv2(img_msg)
    return img


def depth_msg_to_cv2(msg, cv_bridge=CvBridge()):
    img_msg = Image(*slots(msg))
    # convert compressed image message to numpy array
    img = cv_bridge.imgmsg_to_cv2(img_msg)
    img = np.asarray(img, dtype=np.float32)
    return img


def xyz_to_point(xyz):
    point = Point()
    point.x = xyz[0]
    point.y = xyz[1]
    point.z = xyz[2]
    return point


def gridmap_msg_to_numpy(grid_map_msg, layer_name='elevation'):
    # Extract metadata
    W = int(grid_map_msg.info.length_x / grid_map_msg.info.resolution)
    H = int(grid_map_msg.info.length_y / grid_map_msg.info.resolution)
    W = int(W)
    H = int(H)

    # Find the index of the layer
    layer_index = grid_map_msg.layers.index(layer_name)

    # Extract the data for the layer
    grid_map = np.array(grid_map_msg.data[layer_index].data, dtype=np.float32).reshape((H, W))

    # Correct indexing based on start indices
    outer_start_index = grid_map_msg.outer_start_index
    inner_start_index = grid_map_msg.inner_start_index
    # Shift the data using outer and inner indices to correct its position
    grid_map = np.roll(grid_map, shift=-outer_start_index, axis=1)
    grid_map = np.roll(grid_map, shift=-inner_start_index, axis=0)

    grid_map = rotate(grid_map, 180)

    return grid_map


def pose_to_matrix(pose: Pose) -> np.ndarray:
    # Extract translation
    t = np.array([pose.position.x, pose.position.y, pose.position.z])

    # Extract rotation as quaternion and convert to rotation matrix
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    rot = Rotation.from_quat(q).as_matrix()  # 3x3 rotation matrix

    # Construct homogeneous 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = t
    return T


def poses_to_path(poses, stamp=None, frame_id=None):
    assert isinstance(poses, np.ndarray) or isinstance(poses, torch.Tensor)
    assert poses.ndim == 3 or poses.ndim == 2
    n_poses = poses.shape[0]
    if poses.ndim == 3:
        assert poses.shape == (n_poses, 4, 4)
    elif poses.ndim == 2:
        assert poses.shape == (n_poses, 7)
    if stamp is None:
        stamp = rclpy.time.Time()
    if frame_id is None:
        frame_id = 'base_link'
    path = Path()
    path.header.stamp = stamp
    path.header.frame_id = frame_id
    for i in range(poses.shape[0]):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = frame_id
        if poses.ndim == 3:
            tr = pose[i, :3, 3]
            q = Rotation.from_matrix(pose[i, :3, :3]).as_quat()
            pose.pose.position.x = tr[0]
            pose.pose.position.y = tr[1]
            pose.pose.position.z = tr[2]
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
        elif poses.ndim == 2:
            pose.pose.position.x = poses[i, 0]
            pose.pose.position.y = poses[i, 1]
            pose.pose.position.z = poses[i, 2]
            pose.pose.orientation.x = poses[i, 3]
            pose.pose.orientation.y = poses[i, 4]
            pose.pose.orientation.z = poses[i, 5]
            pose.pose.orientation.w = poses[i, 6]
        path.poses.append(pose)
    return path