#!/usr/bin/env python3

import os
from copy import copy

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from PIL import Image as PILImage

import rclpy.time
from monoforce.utils import read_yaml
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.terrain_encoder.utils import sample_augmentation, img_transform, normalize_img

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.impl.logging_severity import LoggingSeverity
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from grid_map_msgs.msg import GridMap
from monoforce.ros import height_map_to_gridmap_msg
import tf2_ros


monoforce_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../'))


class TerrainEncoder(Node):

    def __init__(self):
        super().__init__('terrain_encoder')
        self.declare_parameter('weights', os.path.join(monoforce_path, 'config/weights/lss/val.pth'))
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('fixed_frame', 'odom')
        self.declare_parameter('img_topics', ['/camera/image_raw'])
        self.declare_parameter('camera_info_topics', ['/camera/camera_info'])
        self.declare_parameter('max_msgs_delay', 0.1)
        self.declare_parameter('max_age', 0.2)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._logger.set_level(LoggingSeverity.DEBUG)

        self.lss_cfg = read_yaml(os.path.join(monoforce_path, 'config/lss_cfg.yaml'))
        weights = self.get_parameter('weights').get_parameter_value().string_value
        self._logger.info(f'Loading LSS model from {weights}')
        if not os.path.exists(weights):
            self._logger.error(f'Model weights file {weights} does not exist. Using random weights.')
        self.model = LiftSplatShoot(self.lss_cfg['grid_conf'], self.lss_cfg['data_aug_conf']).from_pretrained(weights)
        self.model.to(self.device)
        self.model.eval()

        self.robot_frame = self.get_parameter('robot_frame').get_parameter_value().string_value
        self.fixed_frame = self.get_parameter('fixed_frame').get_parameter_value().string_value

        self.img_topics = self.get_parameter('img_topics').get_parameter_value().string_array_value
        self.camera_info_topics = self.get_parameter('camera_info_topics').get_parameter_value().string_array_value
        assert len(self.img_topics) == len(self.camera_info_topics)

        self.cv_bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.time.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.max_msgs_delay = self.get_parameter('max_msgs_delay').get_parameter_value().double_value
        self.max_age = self.get_parameter('max_age').get_parameter_value().double_value

        # grid map publisher
        self.gridmap_pub = self.create_publisher(GridMap, '/terrain/grid_map', 10)

    def spin(self):
        try:
            rclpy.spin(self)
        except (KeyboardInterrupt, ExternalShutdownException):
            self.get_logger().info('Keyboard interrupt, shutting down...')
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def start(self):
        self.subs = []
        for topic in self.img_topics:
            self._logger.info(f'Subscribing to {topic}')
            self.subs.append(Subscriber(self, CompressedImage, topic))
        for topic in self.camera_info_topics:
            self._logger.info(f'Subscribing to {topic}')
            self.subs.append(Subscriber(self, CameraInfo, topic))
        self.sync = ApproximateTimeSynchronizer(self.subs, queue_size=1, slop=self.max_msgs_delay)
        self.sync.registerCallback(self.callback)

    def callback(self, *msgs):
        self._logger.debug('Received %d messages' % len(msgs))
        # if a message is stale, do not process it
        t_now = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec / 1e9
        t_msg = msgs[0].header.stamp.sec + msgs[0].header.stamp.nanosec / 1e9
        dt = abs(t_now - t_msg)
        if dt > self.max_age:
            self._logger.warning(f'Message is too old (time diff: {dt:.3f} s), skipping...')
        else:
            # process the messages
            self.proc(*msgs)

    @torch.inference_mode()
    def proc(self, *msgs):
        n = len(msgs)
        assert n % 2 == 0
        for i in range(n // 2):
            assert isinstance(msgs[i], CompressedImage), 'First %d messages must be Image' % (n // 2)
            assert isinstance(msgs[i + n // 2], CameraInfo), 'Last %d messages must be CameraInfo' % (n // 2)
            assert msgs[i].header.frame_id == msgs[i + n // 2].header.frame_id, \
                'Image and CameraInfo messages must have the same frame_id'
        # preprocessing
        img_msgs = msgs[:n // 2]
        info_msgs = msgs[n // 2:]
        inputs = self.get_lss_inputs(img_msgs, info_msgs)
        inputs = [i.to(self.device) for i in inputs]

        # model inference
        out = self.model(*inputs)
        height_terrain, friction = out['terrain'], out['friction']
        self._logger.info('Predicted height map shape: %s' % str(height_terrain.shape))

        # publish height map as grid map
        stamp = msgs[0].header.stamp
        height = height_terrain.squeeze().cpu().numpy()
        grid_msg = height_map_to_gridmap_msg(height, grid_res=self.lss_cfg['grid_conf']['xbound'][2],
                                             xyz=np.array([0., 0., 0.]), q=np.array([0., 0., 0., 1.]))
        grid_msg.header.stamp = stamp
        grid_msg.header.frame_id = self.robot_frame
        self.gridmap_pub.publish(grid_msg)

    def get_transform(self, from_frame, to_frame, time=None):
        """Retrieve a transformation matrix between two frames using TF2."""
        if time is None:
            time = rclpy.time.Time()
        timeout = rclpy.time.Duration(seconds=1.0)
        try:
            tf = self.tf_buffer.lookup_transform(to_frame, from_frame,
                                                 time=time, timeout=timeout)
        except Exception as ex:
            tf = self.tf_buffer.lookup_transform(to_frame, from_frame,
                                                 time=rclpy.time.Time(), timeout=timeout)
            self._logger.warning(
                f"Could not find transform from {from_frame} to {to_frame} at time {time}, using latest available: {tf}"
            )
        # Convert TF2 transform message to a 4x4 transformation matrix
        translation = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
        qaut = [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
        T = np.eye(4)
        R = Rotation.from_quat(qaut).as_matrix()
        T[:3, 3] = translation
        T[:3, :3] = R
        return T

    def get_cam_calib_from_info_msg(self, msg):
        """
        Get camera calibration parameters from CameraInfo message.
        :param msg: CameraInfo message
        :return: E - extrinsics (4x4),
                 K - intrinsics (3x3),
                 D - distortion coefficients (5,)
        """
        assert isinstance(msg, CameraInfo)

        # get camera extrinsics
        E = self.get_transform(from_frame=msg.header.frame_id,
                               to_frame=self.robot_frame,
                               time=msg.header.stamp)
        K = np.array(msg.k).reshape((3, 3))
        D = np.array(msg.d)

        return E, K, D

    def preprocess_img(self, img):
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        # preprocessing parameters (resize, crop)
        lss_cfg = copy(self.lss_cfg)
        lss_cfg['data_aug_conf']['H'], lss_cfg['data_aug_conf']['W'] = img.shape[:2]
        resize, resize_dims, crop, flip, rotate = sample_augmentation(lss_cfg, is_train=False)
        img, post_rot2, post_tran2 = img_transform(PILImage.fromarray(img), post_rot, post_tran,
                                                   resize=resize,
                                                   resize_dims=resize_dims,
                                                   crop=crop,
                                                   flip=False,
                                                   rotate=0)
        # normalize image (subtraction of mean and division by std)
        img = normalize_img(img)

        # for convenience, make augmentation matrices 3x3
        post_tran = torch.zeros(3, dtype=torch.float32)
        post_rot = torch.eye(3, dtype=torch.float32)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2

        return img, post_rot, post_tran

    def get_lss_inputs(self, img_msgs, info_msgs):
        """
        Get inputs for LSS model from image and camera info messages.
        """
        assert len(img_msgs) == len(info_msgs)

        robot_pose = self.get_transform(from_frame=self.robot_frame,
                                        to_frame=self.fixed_frame,
                                        time=img_msgs[0].header.stamp)
        roll, pitch, yaw = Rotation.from_matrix(robot_pose[:3, :3]).as_euler('xyz')
        R = Rotation.from_euler('xyz', [roll, pitch, 0]).as_matrix()

        imgs = []
        post_rots = []
        post_trans = []
        intriniscs = []
        cams_to_robot = []
        for cam_i, (img_msg, info_msg) in enumerate(zip(img_msgs, info_msgs)):
            assert isinstance(img_msg, CompressedImage)
            assert isinstance(info_msg, CameraInfo)

            img = self.cv_bridge.compressed_imgmsg_to_cv2(img_msg)
            self._logger.debug('Input image shape: %s' % str(img.shape))
            # BGR to RGB
            img = img[..., ::-1]
            E, K, D = self.get_cam_calib_from_info_msg(info_msg)

            # extrinsics relative to gravity-aligned frame
            E[:3, :3] = R @ E[:3, :3]

            img, post_rot, post_tran = self.preprocess_img(img)
            imgs.append(img)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            intriniscs.append(K)
            cams_to_robot.append(E)

        # to arrays
        imgs = np.stack(imgs)
        post_rots = np.stack(post_rots)
        post_trans = np.stack(post_trans)
        intrins = np.stack(intriniscs)
        cams_to_robot = np.stack(cams_to_robot)
        rots, trans = cams_to_robot[:, :3, :3], cams_to_robot[:, :3, 3]
        self._logger.debug('Preprocessed image shape: %s' % str(imgs.shape))

        inputs = [imgs, rots, trans, intrins, post_rots, post_trans]
        inputs = [torch.as_tensor(i[np.newaxis], dtype=torch.float32) for i in inputs]

        return inputs


def main(args=None):
    rclpy.init(args=args)
    node = TerrainEncoder()
    node.start()
    node.spin()


if __name__ == '__main__':
    main()
