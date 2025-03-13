#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.img_pub = self.create_publisher(Image, 'image_raw', 10)
        self.caminfo_pub = self.create_publisher(CameraInfo, 'camera_info', 10)
        self.timer = self.create_timer(0.1, self.publish_cam)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use webcam
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def publish_cam(self):
        ret, frame = self.read_image()
        if not ret:
            self.get_logger().error("Failed to capture image")
            return

        # get an image message
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "camera_frame"

        # camera info message
        caminfo_msg = CameraInfo()
        caminfo_msg.header = img_msg.header
        caminfo_msg.height = frame.shape[0]
        caminfo_msg.width = frame.shape[1]
        caminfo_msg.distortion_model = "plumb_bob"
        caminfo_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        caminfo_msg.k = [615.0, 0.0, 320.0, 0.0, 615.0, 240.0, 0.0, 0.0, 1.0]
        caminfo_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        caminfo_msg.p = [615.0, 0.0, 320.0, 0.0, 0.0, 615.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        self.img_pub.publish(img_msg)
        self.caminfo_pub.publish(caminfo_msg)
        self.get_logger().debug("Published image and K messages")

        # publish tf message: identity transform
        t = tf2_ros.TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "camera_frame"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)


    def read_image(self):
        ret, frame = self.cap.read()
        return ret, frame


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
