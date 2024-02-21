from __future__ import division, absolute_import, print_function
import os
import torch
import numpy as np
import yaml
from scipy.ndimage import rotate
from tqdm import tqdm
from cv_bridge import CvBridge
from jsk_recognition_msgs.msg import BoundingBox
from monoforce.utils import slots, timing
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, PoseArray, TransformStamped
from tf.transformations import quaternion_from_matrix, euler_from_quaternion, quaternion_matrix
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from grid_map_msgs.msg import GridMap
from sensor_msgs.msg import PointCloud2, CompressedImage, Image
from ros_numpy import msgify, numpify
from numpy.lib.recfunctions import unstructured_to_structured
from tf2_ros import BufferCore, TransformException
from rosbag import Bag, ROSBagException
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
    x, y = np.meshgrid(np.arange(-H//2, H//2), np.arange(-W//2, W//2))
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


def get_topic_types(bag):
    return {k: v.msg_type for k, v in bag.get_type_and_topic_info().topics.items()}


@timing
def get_closest_msg(bag, topic, time_moment, time_window=1.0,
                    max_time_diff=0.5, max_time_window=10.0,
                    verbose=False):
    assert isinstance(bag, Bag)
    assert isinstance(topic, str)
    assert isinstance(time_moment, float)
    assert isinstance(time_window, float) and time_window > 0
    assert isinstance(max_time_diff, float) and max_time_diff > 0

    if time_window > max_time_window:
        raise Exception('Time window is too large: %.3f [sec]' % time_window)

    stamps_in_window = []
    msgs = []
    tl = max(time_moment - time_window / 2., 0)
    tr = time_moment + time_window / 2.
    for topic, msg, stamp in bag.read_messages(topics=[topic],
                                               start_time=rospy.Time.from_seconds(tl),
                                               end_time=rospy.Time.from_seconds(tr)):
        stamps_in_window.append(stamp.to_sec())
        msgs.append(msg)

    if len(stamps_in_window) == 0:
        # # raise Exception('No image messages in window')
        print('No image messages in window for cloud time %.3f [sec] and topic "%s"' % (time_moment, topic))
        return None, None

    time_diffs = np.abs(np.array(stamps_in_window) - time_moment)
    i_min = np.argmin(time_diffs)
    msg = msgs[i_min]
    msg_stamp = stamps_in_window[i_min]

    time_diff = np.min(time_diffs)
    if verbose:
        print('Got the closest message with time difference: %.3f [sec]' % time_diff)
    assert time_diff < max_time_diff, 'Time difference is too large: %.3f [sec]' % time_diff

    return msg, msg_stamp


def get_cams_robot_transformations(bag_path, camera_topics, robot_frame, tf_buffer, save=True, output_path=None):
    dir = os.path.dirname(bag_path)
    if output_path is None:
        output_path_pattern = '{dir}/{name}/calibration/transformations.yaml'
        output_path = output_path_pattern.format(dir=dir, name=bag_path.split('/')[-1].split('.')[0])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    Trs = []
    with open(output_path, 'w') as f:
        try:
            with Bag(bag_path, 'r') as bag:
                for cam_topic in camera_topics:
                    img_msg = None
                    for topic, msg, stamp in bag.read_messages(topics=[cam_topic]):
                        if topic == cam_topic:
                            img_msg = msg
                            # print('Got image msg at %.3f s' % img_msg.header.stamp.to_sec())

                        if img_msg is None:
                            # print('No image msg read from %s' % bag_path)
                            continue

                        print('Got image msg at %.3f s' % img_msg.header.stamp.to_sec())

                        # find transformation between camera and lidar
                        try:
                            robot_to_camera = tf_buffer.lookup_transform_core(img_msg.header.frame_id,
                                                                              robot_frame,
                                                                              img_msg.header.stamp)
                        except TransformException as ex:
                            print('Could not transform from %s to %s at %.3f s.' %
                                  (robot_frame, img_msg.header.frame_id, img_msg.header.stamp.to_sec()))
                            continue
                        print('Got transformation from %s to %s at %.3f s' % (robot_frame,
                                                                              img_msg.header.frame_id,
                                                                              img_msg.header.stamp.to_sec()))
                        Tr = numpify(robot_to_camera.transform)
                        print('Tr:\n', Tr)
                        Trs.append(Tr)

                        # save transformation to output_path_patern yaml file
                        if save:
                            camera = cam_topic.split('/')[1]
                            print('Saving to %s' % output_path)

                            f.write('T_{robot_frame}__{camera}:\n'.format(robot_frame=robot_frame, camera=camera))
                            f.write('  rows: 4\n')
                            f.write('  cols: 4\n')
                            f.write('  data: [%s]\n' % ', '.join(['%.3f' % x for x in Tr.reshape(-1)]))
                        break
        except ROSBagException as ex:
            print('Could not read %s: %s' % (bag_path, ex))

        f.close()
    return Trs


def get_cams_lidar_transformations(bag_path, camera_topics, lidar_frame, tf_buffer, save=True, output_path=None):
    if isinstance(bag_path, list):
        assert len(bag_path) > 0, 'No bag files provided'
        bag_path = bag_path[0]
        print('Using the first bag file from:', bag_path)

    dir = os.path.dirname(bag_path)
    if output_path is None:
        output_path_pattern = '{dir}/{name}/calibration/transformations.yaml'
        output_path = output_path_pattern.format(dir=dir, name=bag_path.split('/')[-1].split('.')[0])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        try:
            with Bag(bag_path, 'r') as bag:
                for cam_topic in camera_topics:
                    for topic, img_msg, stamp in bag.read_messages(topics=[cam_topic]):
                        # find transformation between camera and lidar
                        try:
                            lidar_to_camera = tf_buffer.lookup_transform_core(img_msg.header.frame_id,
                                                                              lidar_frame,
                                                                              img_msg.header.stamp)
                        except TransformException as ex:
                            print('Could not transform from %s to %s at %.3f s.' %
                                  (lidar_frame, img_msg.header.frame_id, img_msg.header.stamp.to_sec()))
                            continue
                        print('Got transformation from %s to %s at %.3f s' % (lidar_frame,
                                                                              img_msg.header.frame_id,
                                                                              img_msg.header.stamp.to_sec()))
                        Tr = numpify(lidar_to_camera.transform)
                        print('Tr:\n', Tr)

                        # save transformation to yaml file
                        if save:
                            camera = cam_topic.split('/')[1]
                            print('Saving to %s' % output_path)

                            f.write(f'T_{lidar_frame}__{camera}:\n')
                            f.write('  rows: 4\n')
                            f.write('  cols: 4\n')
                            f.write('  data: [%s]\n' % ', '.join(['%.3f' % x for x in Tr.reshape(-1)]))
                        break
        except ROSBagException as ex:
            print('Could not read %s: %s' % (bag_path, ex))

        f.close()


def get_camera_infos(bag_path, camera_info_topics, save=True, output_path=None):
    """
    Read camera intrinsics and distortion coefficients from bag file
    """
    if isinstance(bag_path, list):
        assert len(bag_path) > 0, 'No bag files provided'
        bag_path = bag_path[0]
        print('Using the first bag file from:', bag_path)

    if output_path is None:
        dir = os.path.dirname(bag_path)
        output_path_pattern = '{dir}/{name}/calibration/cameras/'
        output_path = output_path_pattern.format(dir=dir, name=bag_path.split('/')[-1].split('.')[0])
    os.makedirs(output_path, exist_ok=True)

    Ks = []
    Ds = []
    try:
        with Bag(bag_path, 'r') as bag:
            for caminfo_topic in camera_info_topics:
                for topic, msg, stamp in bag.read_messages(topics=[caminfo_topic]):
                    K = np.asarray(msg.K).reshape(3, 3)
                    D = np.asarray(msg.D)
                    print('Read the camera params from "%s" for intrinsics topic "%s"' % (bag_path, topic))
                    print('K:\n', K)
                    print('D:\n', D)
                    Ks.append(K)
                    Ds.append(D)

                    # save intinsics and distortion coeffs to output_path yaml file
                    if save:
                        camera = topic.split('/')[1]
                        output_path_cam = os.path.join(output_path, '%s.yaml' % camera)
                        os.makedirs(os.path.dirname(output_path_cam), exist_ok=True)
                        print('Saving to %s' % output_path_cam)
                        with open(output_path_cam, 'w') as f:
                            f.write('image_width: %d\n' % msg.width)
                            f.write('image_height: %d\n' % msg.height)
                            f.write('camera_name: %s\n' % camera)
                            f.write('camera_matrix:\n')
                            f.write('  rows: 3\n')
                            f.write('  cols: 3\n')
                            f.write('  data: [%s]\n' % ', '.join(['%.12f' % x for x in K.reshape(-1)]))
                            f.write('distortion_model: %s\n' % msg.distortion_model)
                            f.write('distortion_coefficients:\n')
                            f.write('  rows: 1\n')
                            f.write('  cols: %d\n' % len(D))
                            f.write('  data: [%s]\n' % ', '.join(['%.12f' % x for x in D]))
                        f.close()
                    break
    except ROSBagException as ex:
        print('Could not read %s: %s' % (bag_path, ex))
    return Ks, Ds


def append_transformation(bag_paths, source_frame='base_link', target_frame='base_footprint', save=True, tf_buffer=None,
                          matrix_name=None):
    """
    Append transformation from source_frame to target_frame to the yaml file
    """
    assert isinstance(bag_paths, list)
    assert len(bag_paths) > 0, 'No bag files provided'

    if tf_buffer is None:
        tf_buffer = load_tf_buffer(bag_paths, tf_topics=['/tf_static'])
    try:
        transform = tf_buffer.lookup_transform_core(source_frame, target_frame, rospy.Time())
        Tr = numpify(transform.transform)
    except TransformException as ex:
        print('Could not find transformation from %s to %s.' % (source_frame, target_frame))
        return
    print('Transformation from %s to %s:' % (source_frame, target_frame))
    print(Tr)

    if save:
        bag_path = bag_paths[0]
        output_path_pattern = '{dir}/{name}/calibration/transformations.yaml'
        output_path = output_path_pattern.format(dir=os.path.dirname(bag_path),
                                                 name=os.path.basename(bag_path).replace('.bag', ''))

        if matrix_name is None:
            matrix_name = f'T_{source_frame}__{target_frame}'
        new_yaml_data_dict = {matrix_name:
                                  {'rows': 4,
                                   'cols': 4,
                                   'data': ['%.3f' % x for x in Tr.reshape(-1)]}}

        with open(output_path, 'r') as yamlfile:
            print('Updating yaml file: %s' % output_path)
            cur_yaml = yaml.load(yamlfile, Loader=yaml.FullLoader)
            cur_yaml.update(new_yaml_data_dict)

        with open(output_path, 'w') as yamlfile:
            yaml.safe_dump(cur_yaml, yamlfile)  # Also note the safe_dump
