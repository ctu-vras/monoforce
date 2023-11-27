import torch
import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm
from jsk_recognition_msgs.msg import BoundingBox
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, PoseArray, TransformStamped
from tf.transformations import quaternion_from_matrix, euler_from_quaternion, quaternion_matrix
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from grid_map_msgs.msg import GridMap
from sensor_msgs.msg import PointCloud2
from ros_numpy import msgify
from numpy.lib.recfunctions import unstructured_to_structured
from tf2_ros import BufferCore
from rosbag import Bag, ROSBagException
import rospy
from visualization_msgs.msg import Marker


def height_map_to_gridmap_msg(height, grid_res,
                              xyz=np.array([0, 0, 0]), q=np.array([0., 0., 0., 1.])):
    assert isinstance(height, np.ndarray)
    assert height.ndim == 2

    H, W = height.shape
    # rotate height map by 180 degrees
    height = rotate(height, 180)

    map = GridMap()
    map.info.resolution = grid_res
    map.info.length_x = W * grid_res
    map.info.length_y = H * grid_res
    map.info.pose.position = msgify(Point, xyz)
    map.info.pose.orientation = msgify(Quaternion, q)

    map.layers.append('elevation')

    height_array = Float32MultiArray()
    height_array.layout.dim.append(MultiArrayDimension())
    height_array.layout.dim.append(MultiArrayDimension())
    height_array.layout.dim[0].label = 'column_index'
    height_array.layout.dim[0].size = H
    height_array.layout.dim[0].stride = H * W
    height_array.layout.dim[1].label = 'row_index'
    height_array.layout.dim[1].size = W
    height_array.layout.dim[1].stride = W
    height_array.data = height.ravel().tolist()
    map.data.append(height_array)

    return map

def height_map_to_point_cloud_msg(height, grid_res, xyz=np.asarray([0., 0., 0.]), q=np.asarray([0., 0., 0., 1.])):
    assert isinstance(height, np.ndarray)
    assert height.ndim == 2
    H, W = height.shape
    n_pts = H * W
    x, y = np.meshgrid(np.arange(-H//2, H//2), np.arange(-W//2, W//2))
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


def to_tf(pose, frame_id, child_frame_id, stamp=None):
    assert pose.shape == (4, 4)
    if stamp is None:
        stamp = rospy.Time.now()

    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id
    t.transform.translation.x = pose[0, 3]
    t.transform.translation.y = pose[1, 3]
    t.transform.translation.z = pose[2, 3]
    q = quaternion_from_matrix(pose)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    return t

def to_cloud_msg(cloud, stamp=None, frame_id=None, fields=None):
    assert isinstance(cloud, np.ndarray)
    assert cloud.shape[1] >= 3
    if fields is None:
        fields = ['x', 'y', 'z']
    # https://answers.ros.org/question/197309/rviz-does-not-display-pointcloud2-if-encoding-not-float32/
    cloud = np.asarray(cloud, dtype=np.float32)
    cloud_struct = unstructured_to_structured(cloud, names=fields)
    return msgify(PointCloud2, cloud_struct, stamp=stamp, frame_id=frame_id)

def to_pose_array(poses, stamp=None, frame_id=None):
    assert isinstance(poses, np.ndarray) or isinstance(poses, torch.Tensor)
    assert poses.shape[1:] == (4, 4)
    pose_array = PoseArray()
    pose_array.header.stamp = stamp
    pose_array.header.frame_id = frame_id
    for i in range(poses.shape[0]):
        pose = msgify(Pose, poses[i])
        pose_array.poses.append(pose)
    return pose_array

def to_path(poses, stamp=None, frame_id=None):
    assert isinstance(poses, np.ndarray) or isinstance(poses, torch.Tensor)
    assert poses.shape[1:] == (4, 4)
    if stamp is None:
        stamp = rospy.Time.now()
    if frame_id is None:
        frame_id = 'base_link'
    path = Path()
    path.header.stamp = stamp
    path.header.frame_id = frame_id
    for i in range(poses.shape[0]):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = frame_id
        pose.pose = msgify(Pose, poses[i])
        path.poses.append(pose)
    return path

def to_box_msg(pose, size, stamp=None, frame_id=None):
    assert isinstance(pose, np.ndarray) or isinstance(pose, torch.Tensor)
    assert pose.shape == (4, 4)
    assert isinstance(size, np.ndarray) or isinstance(size, torch.Tensor)
    assert size.shape == (3,)
    box = BoundingBox()
    box.header.stamp = stamp
    box.header.frame_id = frame_id
    box.pose.position.x = pose[0, 3]
    box.pose.position.y = pose[1, 3]
    box.pose.position.z = pose[2, 3]
    q = quaternion_from_matrix(pose)
    box.pose.orientation.x = q[0]
    box.pose.orientation.y = q[1]
    box.pose.orientation.z = q[2]
    box.pose.orientation.w = q[3]
    box.dimensions.x = size[0]
    box.dimensions.y = size[1]
    box.dimensions.z = size[2]
    return box

def to_marker(poses, color=None):
    assert isinstance(poses, np.ndarray) or isinstance(poses, torch.Tensor)
    assert poses.shape[1:] == (4, 4)
    marker = Marker()
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.1
    if color is not None:
        assert len(color) == 3
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
    marker.color.a = 1.0
    for t in range(poses.shape[0]):
        p = poses[t]
        marker.points.append(msgify(Pose, p).position)
    return marker
