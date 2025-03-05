import cv2
import numpy as np


__all__ = [
    'undistort_image',
    'project_cloud_to_image'
]


def undistort_image(image, camera_matrix, distortion_coeffs):
    """
    Undistort image using camera matrix and distortion coefficients.
    Reference: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    # Get the image size
    h, w = image.shape[:2]

    # Generate new camera matrix from parameters
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))

    # Undistort the image
    undistorted_image = cv2.undistort(image, camera_matrix, distortion_coeffs, None, new_camera_matrix)

    # Crop the image (optional)
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y + h, x:x + w]

    return undistorted_image, new_camera_matrix


def project_cloud_to_image(points, img, K, return_mask=False, debug=False):
    """Project 3D points to image plane."""
    assert points.shape[1] == 3
    assert K.shape[0] == K.shape[1]
    assert K.shape[0] == 3

    # Project points to image plane.
    points_uv = points.copy()
    points_uv = (K[:3, :3] @ points_uv.T).T
    # Perform perspective division.
    points_uv[:, :2] /= points_uv[:, 2:3]
    # Round to the nearest pixel.
    points_uv[:, 2] = np.round(points_uv[:, 2]).astype(np.int32)

    # Filter out points outside the image.
    h, w = img.shape[:2]
    fov_mask = (points_uv[:, 0] > 1) & (points_uv[:, 0] < w - 1) & \
               (points_uv[:, 1] > 1) & (points_uv[:, 1] < h - 1) & \
               (points_uv[:, 2] > 0)

    # colorize the cloud with colors from the image
    colors = np.zeros_like(points)
    colors[fov_mask] = img[points_uv[fov_mask, 1].astype(np.int32), points_uv[fov_mask, 0].astype(np.int32)]

    if debug:
        # show image with projected points as circles
        img_vis = img.copy()
        for i in range(points.shape[0]):
            if fov_mask[i]:
                cv2.circle(img_vis, (int(points_uv[i, 0]), int(points_uv[i, 1])), 2, (0, 0, 255), -1)
        # img_vis = cv2.resize(img_vis, (1280, 960))
        cv2.imshow('img', img_vis)
        cv2.waitKey(0)

    if return_mask:
        return points[fov_mask], colors, fov_mask

    return points[fov_mask], colors
