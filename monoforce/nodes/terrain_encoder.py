import os
from copy import copy
from threading import RLock
import torch
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
# from grid_map_msgs.msg import GridMap
from monoforce.ros import height_map_to_gridmap_msg
from monoforce.utils import load_calib
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.terrain_encoder.utils import img_transform, normalize_img, sample_augmentation
from sensor_msgs.msg import CompressedImage, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros
from PIL import Image as PILImage
# from ros_numpy import numpify
from scipy.spatial.transform import Rotation

torch.set_default_dtype(torch.float32)
monoforce_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../'))


class TerrainEncoder(Node):
    def __init__(self, lss_cfg: dict):
        super().__init__('terrain_encoder')
        self.lss_cfg = lss_cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.declare_parameter('rate', None)
        self.declare_parameter('weights', os.path.join(monoforce_path, 'config/weights/lss/lss.pt'))
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('fixed_frame', 'odom')
        self.declare_parameter('img_topics', [])
        self.declare_parameter('n_cams', None)
        self.declare_parameter('camera_info_topics', [])
        self.declare_parameter('calib_path', '')
        self.declare_parameter('max_msgs_delay', 0.1)
        self.declare_parameter('max_age', 0.2)

        rate = self.get_parameter('rate').get_parameter_value().double_value
        self.rate = self.create_rate(rate) if rate else None

        weights = self.get_parameter('weights').get_parameter_value().string_value
        self.get_logger().info(f'Loading LSS model from {weights}')
        if not os.path.exists(weights):
            self.get_logger().error(f'Model weights file {weights} does not exist. Using random weights.')
        self.model = LiftSplatShoot(self.lss_cfg['grid_conf'], self.lss_cfg['data_aug_conf']).from_pretrained(weights)
        self.model.to(self.device)
        self.model.eval()

        self.robot_frame = self.get_parameter('robot_frame').get_parameter_value().string_value
        self.fixed_frame = self.get_parameter('fixed_frame').get_parameter_value().string_value

        img_topics = self.get_parameter('img_topics').get_parameter_value().string_array_value
        n_cams = self.get_parameter('n_cams').get_parameter_value().integer_value
        self.img_topics = img_topics[:n_cams] if isinstance(n_cams, int) else img_topics

        camera_info_topics = self.get_parameter('camera_info_topics').get_parameter_value().string_array_value
        self.camera_info_topics = camera_info_topics[:n_cams] if isinstance(n_cams, int) else camera_info_topics

        assert len(self.img_topics) == len(self.camera_info_topics)

        calib_path = self.get_parameter('calib_path').get_parameter_value().string_value
        self.calib = load_calib(calib_path) if calib_path else None
        if self.calib:
            self.get_logger().info(f'Loaded calibration from {calib_path}')

        self.cv_bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.gridmap_pub = self.create_publisher(GridMap, '/grid_map/terrain', 1)

        self.proc_lock = RLock()
        self.max_msgs_delay = self.get_parameter('max_msgs_delay').get_parameter_value().double_value
        self.max_age = self.get_parameter('max_age').get_parameter_value().double_value

    def start(self):
        self.subs = []
        for topic in self.img_topics:
            self.get_logger().info(f'Subscribing to {topic}')
            self.subs.append(Subscriber(self, CompressedImage, topic))
        for topic in self.camera_info_topics:
            self.get_logger().info(f'Subscribing to {topic}')
            self.subs.append(Subscriber(self, CameraInfo, topic))
        self.sync = ApproximateTimeSynchronizer(self.subs, queue_size=1, slop=self.max_msgs_delay)
        self.sync.registerCallback(self.callback)

    def preprocess_img(self, img):
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        lss_cfg = copy(self.lss_cfg)
        lss_cfg['data_aug_conf']['H'], lss_cfg['data_aug_conf']['W'] = img.shape[:2]
        resize, resize_dims, crop, flip, rotate = sample_augmentation(lss_cfg, is_train=False)
        img, post_rot2, post_tran2 = img_transform(PILImage.fromarray(img), post_rot, post_tran, resize=resize,
                                                   resize_dims=resize_dims, crop=crop, flip=False, rotate=0)

        img = normalize_img(img)

        post_tran = torch.zeros(3, dtype=torch.float32)
        post_rot = torch.eye(3, dtype=torch.float32)
        post_tran[:2] = post_tran2
        post_rot[:2, :2] = post_rot2

        return img, post_rot, post_tran

    def get_cam_calib_from_yaml(self, camera, robot_frame='base_link'):
        assert self.calib is not None
        Tr_robot_cam = self.calib['transformations'][camera]
        R = Rotation.from_matrix(np.array(Tr_robot_cam[:3, :3])).as_quat()
        translation = np.array(Tr_robot_cam[:3, 3])
        return translation, R

    def callback(self, *msgs):
        with self.proc_lock:
            try:
                # Extract images and camera infos
                images = [self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8") for msg in
                          msgs[:len(self.img_topics)]]
                camera_infos = msgs[len(self.img_topics):]

                # Transform images
                processed_imgs = []
                for img in images:
                    img, post_rot, post_tran = self.preprocess_img(img)
                    processed_imgs.append(img)

                # Stack images into tensor
                imgs_tensor = torch.stack(processed_imgs, dim=0).to(self.device)

                # Get vehicle pose from TF
                try:
                    trans = self.tf_buffer.lookup_transform(self.fixed_frame, self.robot_frame, rclpy.time.Time())
                    translation = np.array([trans.transform.translation.x, trans.transform.translation.y,
                                            trans.transform.translation.z])
                    rotation = np.array(
                        [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z,
                         trans.transform.rotation.w])
                except Exception as e:
                    self.get_logger().warn(f"TF lookup failed: {str(e)}")
                    return

                # Generate terrain height map
                height_map = self.model(imgs_tensor, [translation, rotation])

                # Publish height map as GridMap
                grid_map_msg = height_map_to_gridmap_msg(height_map, frame_id=self.fixed_frame)
                self.gridmap_pub.publish(grid_map_msg)

            except Exception as e:
                self.get_logger().error(f"Processing error: {str(e)}")

    def shutdown(self):
        self.get_logger().info("Shutting down terrain encoder node.")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    # node = TerrainEncoder(lss_cfg={})  # Replace with actual LSS configuration
    # node.start()
    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     node.shutdown()


if __name__ == '__main__':
    main()
