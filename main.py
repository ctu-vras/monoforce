from fire import Fire
from monoforce import datasets


if __name__ == '__main__':
    Fire({
        'global_cloud': datasets.data.global_cloud_demo,
        'extrinsics': datasets.data.extrinsics_demo,
        'rgb_cloud': datasets.data.vis_rgb_cloud,
        'traversed_hm': datasets.data.traversed_height_map,
        'vis_train_sample': datasets.data.vis_train_sample,
        'hm_weights': datasets.data.vis_hm_weights,
        'hm_from_cloud': datasets.data.vis_estimated_height_map,
        'img_augs': datasets.data.vis_img_augs,
    })
