from fire import Fire
from monoforce import datasets
from monoforce.models import lss, monolayout


if __name__ == '__main__':
    Fire({
        'global_cloud': datasets.data.global_cloud_demo,
        'traversed_cloud': datasets.data.segm_demo,
        'extrinsics': datasets.data.extrinsics_demo,
        'rgb_cloud': datasets.data.vis_rgb_cloud,
        'traversed_hm': datasets.data.traversed_height_map,
        'vis_train_sample': datasets.data.vis_train_sample,
        'hm_weights': datasets.data.vis_hm_weights,
        'hm_from_cloud': datasets.data.vis_estimated_height_map,
        'img_augs': datasets.data.vis_img_augs,

        'lidar_check': lss.explore.lidar_check,
        'cumsum_check': lss.explore.cumsum_check,
        'train_lss_nusc': lss.train.train,
        'eval_lss_iou': lss.explore.eval_model_iou,
        'viz_lss_preds': lss.explore.viz_model_preds,
    })
