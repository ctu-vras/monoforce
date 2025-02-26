METAINFO = {
    "classes": (
        "unlabelled",
        "asphalt",
        "dirt",
        "mud",
        "water",
        "gravel",
        "other-terrain",
        "tree-trunk",
        "tree-foliage",
        "bush",
        "fence",
        "structure",
        "pole",
        "vehicle",
        "rock",
        "log",
        "other-object",
        "sky",
        "grass",
    ),
    "palette": [
        (0, 0, 0),
        (255, 165, 0),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (230, 25, 75),
        (0, 128, 128),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (250, 190, 190),
        (0, 0, 128),
        (128, 128, 128),
    ],
    "cidx": [
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
        14,
        15,
        16,
        17,
        18
    ]
}


def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b))


# # METAINFO['cidx'] = list(range(len(METAINFO['palette'])))
# METAINFO['hex'] = [rgb_to_hex(x[0], x[1], x[2]) for x in METAINFO['palette']]
# METAINFO['hash'] = [str(x[0]) + str(x[1]) + str(x[2]) for x in METAINFO['palette']]
# hex_2_class = {h: c for h, c in zip(METAINFO['hex'], METAINFO['classes'])}
# hex_2_cidx = {h: cidx for h, cidx in zip(METAINFO['hex'], METAINFO['cidx'])}
# class_2_hex = {c: h for h, c in hex_2_class.items()}
# rgb_2_class = {p: c for p, c in zip(METAINFO['palette'], METAINFO['classes'])}
# cidx_2_rgb = {cidx: p for cidx, p in zip(METAINFO['cidx'], METAINFO['palette'])}
# cidx_2_class = {cidx: p for cidx, p in zip(METAINFO['cidx'], METAINFO['classes'])}
# class_2_cidx = {c: cidx for c, cidx in zip(METAINFO['classes'], METAINFO['cidx'])}
#
# hash_2_class = {h: c for h, c in zip(METAINFO['hash'], METAINFO['classes'])}
# hash_2_cidx = {h: c for h, c in zip(METAINFO['hash'], METAINFO['cidx'])}
# class_2_rgb = {c: p for c, p in zip(METAINFO['classes'], METAINFO['palette'])}
