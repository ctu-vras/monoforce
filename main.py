from fire import Fire
from monoforce import datasets


if __name__ == '__main__':
    Fire({
        'global_cloud': datasets.robingas_dataset.global_cloud_demo,
        'traversed_cloud': datasets.robingas_dataset.segm_demo,
        'extrinsics': datasets.robingas_dataset.extrinsics_demo,
        'project_rgb_to_cloud': datasets.robingas_dataset.project_rgb_to_cloud,
        'traversed_hm': datasets.robingas_dataset.traversed_height_map,
        'train_sample': datasets.robingas_dataset.vis_train_sample,
        'hm_weights': datasets.robingas_dataset.vis_hm_weights,
        'hm_from_cloud': datasets.robingas_dataset.vis_estimated_height_map,
        'img_augs': datasets.robingas_dataset.vis_img_augs,
    })
