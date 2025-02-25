from .rough import data_dir
import numpy as np
import os
import open3d as o3d

METAINFO = {
    "classes": (
        "unlabelled",
        "bush",
        "dirt",
        "fence",
        "grass",
        "gravel",
        "log",
        "mud",
        "other-object",
        "other-terrain",
        "rock",
        "sky",
        "structure",
        "tree-foliage",
        "tree-trunk",
        "water",
    ),
    "palette": [
        (0, 0, 0),
        (230, 25, 75),
        (60, 180, 75),
        (0, 128, 128),
        (128, 128, 128),
        (145, 30, 180),
        (128, 128, 0),
        (255, 225, 25),
        (250, 190, 190),
        (70, 240, 240),
        (170, 255, 195),
        (0, 0, 128),
        (170, 110, 40),
        (210, 245, 60),
        (240, 50, 230),
        (0, 130, 200),
    ],
    "cidx": [
        255,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14
    ]
}


def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b))


# METAINFO['cidx'] = list(range(len(METAINFO['palette'])))
METAINFO['hex'] = [rgb_to_hex(x[0], x[1], x[2]) for x in METAINFO['palette']]
METAINFO['hash'] = [str(x[0]) + str(x[1]) + str(x[2]) for x in METAINFO['palette']]
hex_2_class = {h: c for h, c in zip(METAINFO['hex'], METAINFO['classes'])}
hex_2_cidx = {h: cidx for h, cidx in zip(METAINFO['hex'], METAINFO['cidx'])}
class_2_hex = {c: h for h, c in hex_2_class.items()}
rgb_2_class = {p: c for p, c in zip(METAINFO['palette'], METAINFO['classes'])}
cidx_2_rgb = {cidx: p for cidx, p in zip(METAINFO['cidx'], METAINFO['palette'])}
cidx_2_class = {cidx: p for cidx, p in zip(METAINFO['cidx'], METAINFO['classes'])}
class_2_cidx = {c: cidx for c, cidx in zip(METAINFO['classes'], METAINFO['cidx'])}

hash_2_class = {h: c for h, c in zip(METAINFO['hash'], METAINFO['classes'])}
hash_2_cidx = {h: c for h, c in zip(METAINFO['hash'], METAINFO['cidx'])}
class_2_rgb = {c: p for c, p in zip(METAINFO['classes'], METAINFO['palette'])}


def load_pcd(cloud, labels, fpv=False):
    # Load points
    pcd = o3d.geometry.PointCloud()
    points = np.fromfile(cloud, dtype=np.float32).reshape(-1, 3)
    # Load colours
    labels = np.fromfile(labels, dtype=np.int32)
    # Need to remap colours to the output palette
    colors = np.array([list(cidx_2_rgb[x]) for x in labels]) / 255

    if fpv:
        index = points[:, 0] >= 0
        points = points[index]
        colors = colors[index]

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main():
    path = os.path.join(data_dir, "WildScenes")

    # load a point cloud
    seq = "K-01"
    time_stamp = "1624325960.615615100"
    cloud_path = os.path.join(path, "WildScenes3d", seq, "Clouds", time_stamp + ".bin")

    # load a label
    label_path = os.path.join(path, "WildScenes3d", seq, "Labels", time_stamp + ".label")

    pcd = load_pcd(cloud_path, label_path)

    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
