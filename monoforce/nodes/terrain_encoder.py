#!/usr/bin/env python3

import os
import torch

from monoforce.utils import read_yaml
from monoforce.models.terrain_encoder.lss import LiftSplatShoot

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros

torch.set_default_dtype(torch.float32)
monoforce_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../'))


class TerrainEncoder(Node):

    def __init__(self):
        super().__init__('terrain_encoder')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.declare_parameter('weights', os.path.join(monoforce_path, 'config/weights/lss/val.pth'))
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('fixed_frame', 'odom')
        self.declare_parameter('img_topics', ['/image_raw'])
        self.declare_parameter('camera_info_topics', ['/camera_info'])
        self.declare_parameter('max_msgs_delay', 0.1)
        self.declare_parameter('max_age', 0.2)

        self.lss_cfg = read_yaml(os.path.join(monoforce_path, 'config/lss_cfg.yaml'))
        weights = self.get_parameter('weights').get_parameter_value().string_value
        self.get_logger().info(f'Loading LSS model from {weights}')
        if not os.path.exists(weights):
            self.get_logger().error(f'Model weights file {weights} does not exist. Using random weights.')
        self.model = LiftSplatShoot(self.lss_cfg['grid_conf'], self.lss_cfg['data_aug_conf']).from_pretrained(weights)
        self.model.to(self.device)
        self.model.eval()

        self.robot_frame = self.get_parameter('robot_frame').get_parameter_value().string_value
        self.fixed_frame = self.get_parameter('fixed_frame').get_parameter_value().string_value

        self.img_topics = self.get_parameter('img_topics').get_parameter_value().string_array_value
        self.camera_info_topics = self.get_parameter('camera_info_topics').get_parameter_value().string_array_value
        assert len(self.img_topics) == len(self.camera_info_topics)

        self.cv_bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.max_msgs_delay = self.get_parameter('max_msgs_delay').get_parameter_value().double_value
        self.max_age = self.get_parameter('max_age').get_parameter_value().double_value

    def start(self):
        self.subs = []
        for topic in self.img_topics:
            self.get_logger().info(f'Subscribing to {topic}')
            self.subs.append(Subscriber(self, Image, topic))
        for topic in self.camera_info_topics:
            self.get_logger().info(f'Subscribing to {topic}')
            self.subs.append(Subscriber(self, CameraInfo, topic))
        self.sync = ApproximateTimeSynchronizer(self.subs, queue_size=1, slop=self.max_msgs_delay)
        self.sync.registerCallback(self.callback)

    def callback(self, *msgs):
        for msg in msgs:
            self.get_logger().info('I heard: "%s"' % msg.header)


def main(args=None):
    try:
        with rclpy.init(args=args):
            terrain_encoder = TerrainEncoder()
            terrain_encoder.start()

            rclpy.spin(terrain_encoder)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
