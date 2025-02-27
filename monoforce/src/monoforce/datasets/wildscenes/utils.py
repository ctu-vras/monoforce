import logging
import cv2 as cv
import numpy as np
import yaml
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R


logger = logging.getLogger(__file__)


def timestamp_to_bag_time(image_timestamp):
    """Convert from image timestamp to bag (epoch) time.

    Images were saved in s since epoch rounded to 9 decimal places separated by '-' but the leading zeros from the fractional part were stripped. We correct the value by forward-filling zeros
    """
    if image_timestamp.__contains__("."):
        sstr = image_timestamp.split(".")
        padded_string = f"{sstr[0]}.{int(sstr[1]):09d}"
    else:
        sstr = image_timestamp.split("-")
        padded_string = f"{sstr[0]}.{int(sstr[1]):09d}"
    return padded_string


def get_ids_2d(dataset_dir):
    """Get a list of ids from the image and indexLabel subdirectories for a dataset and warn if there are missing ids"""
    im_dir = dataset_dir / "image"
    label_dir = dataset_dir / "indexLabel"
    im_ids = set(i.stem for i in im_dir.glob("*"))
    label_ids = set(i.stem for i in label_dir.glob("*"))
    # Warn for missing ids
    missing_im = label_ids - im_ids
    missing_label = im_ids - label_ids
    if missing_im:
        logger.warning(
            f"IDs {','.join(missing_im)} present in {label_dir} but not in {im_dir}"
        )
    if missing_label:
        logger.warning(
            f"IDs {','.join(missing_label)} present in {im_dir} but not in {label_dir}"
        )
    return label_ids.union(im_ids)


def get_ids_3d(dataset_dir):
    """Get a list of ids from the image and indexLabel subdirectories for a dataset and warn if there are missing ids"""
    im_dir = dataset_dir / "Clouds"
    label_dir = dataset_dir / "Labels"
    im_ids = set(i.stem for i in im_dir.glob("*"))
    label_ids = set(i.stem for i in label_dir.glob("*"))
    # Warn for missing ids
    missing_im = label_ids - im_ids
    missing_label = im_ids - label_ids
    if missing_im:
        logger.warning(
            f"IDs {','.join(missing_im)} present in {label_dir} but not in {im_dir}"
        )
    if missing_label:
        logger.warning(
            f"IDs {','.join(missing_label)} present in {im_dir} but not in {label_dir}"
        )
    return label_ids.union(im_ids)


def convert_ts_to_float(timestamps):
    timestamps = np.array(timestamps).astype(np.longdouble)
    return timestamps


def get_extrinsics(params):
    T = np.eye(4)
    T[:3, 3] = params["translation"]
    quat = params["rotation"]  # qx qy qz qw
    T[:3, :3] = R.from_quat(quat).as_matrix()
    return T


def read_rgb_image(file):
    bgr_img = cv.imread(file)
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    return rgb_img


def get_intrinsics(params):
    K = np.eye(3)
    K[0, 0] = params["K"][0]
    K[1, 1] = params["K"][1]
    K[0, 2] = params["K"][2]
    K[1, 2] = params["K"][3]

    D = np.asarray(params["D"])
    return K, D


def viz_image(img, vizpath=None):
    if vizpath == None:
        plt.switch_backend("agg")
        plt.figure(figsize=(15, 20))
        plt.show()
    else:
        plt.imshow(img)
        plt.savefig(vizpath)