from __future__ import division, absolute_import, print_function
import torch
import numpy as np
from scipy.ndimage import rotate
from cv_bridge import CvBridge
from jsk_recognition_msgs.msg import BoundingBox
from monoforce.utils import slots, timing
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, PoseArray, TransformStamped
from tf.transformations import quaternion_from_matrix, quaternion_matrix
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from grid_map_msgs.msg import GridMap
from sensor_msgs.msg import PointCloud2, CompressedImage, Image
from ros_numpy import msgify, numpify
from numpy.lib.recfunctions import unstructured_to_structured
import rospy
from visualization_msgs.msg import Marker


def height_map_to_gridmap_msg(height, grid_res,
                              xyz=np.array([0, 0, 0]), q=np.array([0., 0., 0., 1.]),
                              mask=None):
    assert isinstance(height, np.ndarray)
    assert height.ndim == 2
    if mask is not None:
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 2
        assert mask.shape == height.shape

    H, W = height.shape
    # rotate height map
    height = height.T
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

    if mask is not None:
        mask = mask.T
        mask = rotate(mask, 180)
        map.layers.append('mask')
        mask_array = Float32MultiArray()
        mask_array.layout.dim.append(MultiArrayDimension())
        mask_array.layout.dim.append(MultiArrayDimension())
        mask_array.layout.dim[0].label = 'column_index'
        mask_array.layout.dim[0].size = H
        mask_array.layout.dim[0].stride = H * W
        mask_array.layout.dim[1].label = 'row_index'
        mask_array.layout.dim[1].size = W
        mask_array.layout.dim[1].stride = W
        mask_array.data = mask.ravel().tolist()
        map.data.append(mask_array)

    return map


def height_map_to_point_cloud_msg(height, grid_res, xyz=np.asarray([0., 0., 0.]), q=np.asarray([0., 0., 0., 1.])):
    assert isinstance(height, np.ndarray)
    assert height.ndim == 2
    H, W = height.shape
    n_pts = H * W
    x, y = np.meshgrid(np.arange(-H // 2, H // 2), np.arange(-W // 2, W // 2))
    x = x.ravel() * grid_res
    y = y.ravel() * grid_res
    z = height.T.ravel()
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
    # normalize quaternion
    q /= np.linalg.norm(q)
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


def poses_to_path(poses, stamp=None, frame_id=None):
    assert isinstance(poses, np.ndarray) or isinstance(poses, torch.Tensor)
    assert poses.ndim == 3 or poses.ndim == 2
    n_poses = poses.shape[0]
    if poses.ndim == 3:
        assert poses.shape == (n_poses, 4, 4)
    elif poses.ndim == 2:
        assert poses.shape == (n_poses, 7)
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
        if poses.ndim == 3:
            pose.pose = msgify(Pose, poses[i])
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

def transform_path(path, pose):
    assert isinstance(path, Path)
    assert isinstance(pose, np.ndarray)
    assert pose.shape == (4, 4)
    for i in range(len(path.poses)):
        path_pose = np.matmul(pose, numpify(path.poses[i].pose))
        path.poses[i].pose = msgify(Pose, path_pose)
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
    marker.scale.x = 0.1
    if color is not None:
        assert len(color) == 3
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
    marker.color.a = 1.0
    for t in range(poses.shape[0]):
        p = poses[t]
        if p.shape == (4, 4):
            marker.points.append(msgify(Pose, p).position)
        elif p.shape == (7,):
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            pose.position.z = p[2]
            # pose.orientation.x = p[3]
            # pose.orientation.y = p[4]
            # pose.orientation.z = p[5]
            # pose.orientation.w = p[6]
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
