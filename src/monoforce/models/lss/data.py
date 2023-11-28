"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torchvision import transforms
import os
import numpy as np
from PIL import ImageFile
# https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/162
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from glob import glob
from copy import deepcopy
from collections import namedtuple

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx, get_nusc_maps
from .tools import get_local_map as get_local_map_poly


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]
        
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


class SegmentationDataMap(NuscData):
    def __init__(self, nusc_maps, *args, **kwargs):
        super(SegmentationDataMap, self).__init__(*args, **kwargs)
        self.nusc_maps = nusc_maps

    def __getitem__(self, index):
        rec = self.ixes[index]

        if self.is_train:
            cams = self.choose_cams()
        else:
            cams = ['CAM_FRONT']
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        # local_map = self.get_local_map(rec, cams)

        cars_binimg = self.get_binimg(rec)
        road_binimg = self.get_local_map_front(rec)
        local_map = torch.cat([road_binimg, cars_binimg], dim=0)

        return imgs, rots, trans, intrins, post_rots, post_trans, local_map

    def get_local_map(self, rec, cams, patch_size=100.0, near_plane=1e-8, render_behind_cam=True):
        scene2map = {}
        for scene in self.nusc.scene:
            log = self.nusc.get('log', scene['log_token'])
            scene2map[scene['name']] = log['location']

        map_name = scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

        camera_channel = cams[0]

        # Check that NuScenesMap was loaded for the correct location.
        scene_record = self.nusc.get('scene', rec['scene_token'])
        log_record = self.nusc.get('log', scene_record['log_token'])
        log_location = log_record['location']

        # Grab the camera image and intrinsics.
        cam_token = rec['data'][camera_channel]
        cam_record = self.nusc.get('sample_data', cam_token)
        cs_record = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])

        # Retrieve the current map
        poserecord = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
        ego_pose = poserecord['translation']
        box_coords = (
            ego_pose[0] - patch_size / 2.,
            ego_pose[1] - patch_size / 2.,
            ego_pose[0] + patch_size / 2.,
            ego_pose[1] + patch_size / 2.,
        )
        # Default layers.
        # layer_names = ['road_segment', 'lane']
        layer_names = ['road_segment', 'lane', 'ped_crossing', 'stop_line', 'carpark_area']

        nusc_map = self.nusc_maps[map_name]
        records_in_patch = nusc_map.get_records_in_patch(box_coords, layer_names, 'intersect')

        cam_frame_points = []
        # Retrieve and render each record.
        for layer_name in layer_names:
            for token in records_in_patch[layer_name]:
                record = nusc_map.get(layer_name, token)

                polygon_tokens = [record['polygon_token']]

                for polygon_token in polygon_tokens:
                    polygon = nusc_map.extract_polygon(polygon_token)

                    # Convert polygon nodes to pointcloud with 0 height.
                    points = np.array(polygon.exterior.xy)
                    points = np.vstack((points, np.zeros((1, points.shape[1]))))

                    # Transform into the ego vehicle frame for the timestamp of the image.
                    points = points - np.array(poserecord['translation']).reshape((-1, 1))
                    points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)

                    # Transform into the camera.
                    points_cam_coord = points - np.array(cs_record['translation']).reshape((-1, 1))
                    points_cam_coord = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points_cam_coord)

                    # Remove points that are partially behind the camera.
                    points_cam = deepcopy(points_cam_coord)
                    depths = points_cam[2, :]
                    behind = depths < near_plane

                    if render_behind_cam:
                        # Perform clipping on polygons that are partially behind the camera.
                        points_cam = NuScenesMapExplorer._clip_points_behind_camera(points_cam, near_plane)
                    elif np.any(behind):
                        # Otherwise ignore any polygon that is partially behind the camera.
                        continue

                    # Ignore polygons with less than 3 points after clipping.
                    if len(points_cam) == 0 or points_cam.shape[1] < 3:
                        continue

                    points = points_cam_coord[[0, 2], :]
                    points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
                    cam_frame_points.append(np.array(points))

        # add drivable area
        static_binimg = np.zeros(np.array(self.nx[:2], dtype=np.int))
        for poly_pts in cam_frame_points:
            pts = (poly_pts[:, :2] - self.bx[:2]) / self.dx[:2]
            pts = np.round(pts).astype(np.int32)
            # pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(static_binimg, [pts], 1.0)
        static_binimg = torch.Tensor(static_binimg).unsqueeze(0)

        # add cars
        dynamic_binimg = np.zeros(np.array(self.nx[:2], dtype=np.int))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            # transform to ego frame
            box.translate(-np.array(poserecord['translation']))
            box.rotate(Quaternion(poserecord['rotation']).inverse)
            # transform to camera frame
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

            pts = box.bottom_corners()[[0, 2]].T
            pts = np.round((pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]).astype(np.int32)
            # pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(dynamic_binimg, [pts], 1.0)
        dynamic_binimg = torch.Tensor(dynamic_binimg).unsqueeze(0)

        return torch.cat([static_binimg, dynamic_binimg], dim=0)

    def get_local_map_front(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])

        scene2map = {}
        for scene in self.nusc.scene:
            log = self.nusc.get('log', scene['log_token'])
            scene2map[scene['name']] = log['location']

        map_name = scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1, 0], rot[0, 0])
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

        poly_names = ['road_segment', 'lane']
        line_names = ['road_divider', 'lane_divider']

        lmap_poly = get_local_map_poly(self.nusc_maps[map_name], center, 100.0, poly_names, line_names)
        local_map = np.zeros(self.nx[:2])
        for name in poly_names:
            for la in lmap_poly[name]:
                pts = (la - self.bx[:2]) / self.dx[:2]
                pts = np.round(pts).astype(np.int32)
                pts[:, [1, 0]] = pts[:, [0, 1]]
                cv2.fillPoly(local_map, [pts], 1.0)

        return torch.Tensor(local_map).unsqueeze(0)


class SegmentationDataMapOSM(SegmentationDataMap):
    def __init__(self, osm_path, *args, **kwargs):
        super(SegmentationDataMapOSM, self).__init__(*args, **kwargs)
        self.occ_map_size = int((self.grid_conf['xbound'][1]-self.grid_conf['xbound'][0])/self.grid_conf['xbound'][2])
        self.osm_path = osm_path
        self.tensor_to_pil = transforms.ToPILImage()

    def __getitem__(self, index):
        rec = self.ixes[index]

        if self.is_train:
            cams = self.choose_cams()
        else:
            cams = ['CAM_FRONT']
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        dynamic_binimg = self.get_binimg(rec)
        # local_map = self.get_local_map(rec, cams)
        static_binimg = self.get_local_map_front(rec)
        local_map = torch.cat([static_binimg, dynamic_binimg], dim=0)

        if self.is_train:
            # input for static (road) objects discriminator
            static_discr_input = self.process_discr(self.get_osm(self.osm_path), self.occ_map_size)
            static_discr_input = torch.transpose(torch.Tensor(static_discr_input), 2, 0)
            # static_discr_input = torch.transpose(static_discr_input, 1, 2)

            # input for dynamic (cars) objects discriminator
            dynamic_binimg_pil = self.tensor_to_pil(dynamic_binimg)
            dynamic_discr_input = self.process_discr(dynamic_binimg_pil, self.occ_map_size)
            dynamic_discr_input = torch.transpose(torch.Tensor(dynamic_discr_input), 2, 0)
            dynamic_discr_input = torch.transpose(dynamic_discr_input, 1, 2)

            discr_input = torch.cat([static_discr_input, dynamic_discr_input], dim=0)
            return imgs, rots, trans, intrins, post_rots, post_trans, local_map, discr_input
        else:
            return imgs, rots, trans, intrins, post_rots, post_trans, local_map

    @staticmethod
    def get_osm(root_dir):
        # get_osm_path
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)
        with open(osm_path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB').transpose(Image.ROTATE_90)

    def process_discr(self, topview, size):
        topview = self.resize_topview(topview, size)
        topview_n = np.zeros((size, size, 1))
        topview_n[topview == 255, 0] = 1.
        return topview_n

    @staticmethod
    def resize_topview(topview, size):
        topview = topview.convert("1")
        topview = topview.resize((size, size), Image.NEAREST)
        topview = topview.convert("L")
        topview = np.array(topview)
        return topview


class KITTI360_Map(torch.utils.data.Dataset):
    # a label and all meta information
    Label = namedtuple('Label', [

        'name',  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class

        'id',  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.

        'kittiId',  # An integer ID that is associated with this label for KITTI-360
        # NOT FOR RELEASING

        'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!

        'category',  # The name of the category that this label belongs to

        'categoryId',  # The ID of this category. Used to create ground truth images
        # on category level.

        'hasInstances',  # Whether this label distinguishes between single instances or not

        'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not

        'color',  # The color of this label
    ])

    LABELS = [
        #       name                     id    kittiId,    trainId   category            catId     hasInstances   ignoreInEval   color
        Label('unlabeled', 0, -1, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('ego vehicle', 1, -1, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('rectification border', 2, -1, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('out of roi', 3, -1, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('static', 4, -1, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('dynamic', 5, -1, 255, 'void', 0, False, True, (111, 74, 0)),
        Label('ground', 6, -1, 255, 'void', 0, False, True, (81, 0, 81)),
        Label('road', 7, 1, 0, 'flat', 1, False, False, (128, 64, 128)),
        Label('sidewalk', 8, 3, 1, 'flat', 1, False, False, (244, 35, 232)),
        Label('parking', 9, 2, 255, 'flat', 1, False, True, (250, 170, 160)),
        Label('rail track', 10, 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        Label('building', 11, 11, 2, 'construction', 2, True, False, (70, 70, 70)),
        Label('wall', 12, 7, 3, 'construction', 2, False, False, (102, 102, 156)),
        Label('fence', 13, 8, 4, 'construction', 2, False, False, (190, 153, 153)),
        Label('guard rail', 14, 30, 255, 'construction', 2, False, True, (180, 165, 180)),
        Label('bridge', 15, 31, 255, 'construction', 2, False, True, (150, 100, 100)),
        Label('tunnel', 16, 32, 255, 'construction', 2, False, True, (150, 120, 90)),
        Label('pole', 17, 21, 5, 'object', 3, True, False, (153, 153, 153)),
        Label('polegroup', 18, -1, 255, 'object', 3, False, True, (153, 153, 153)),
        Label('traffic light', 19, 23, 6, 'object', 3, True, False, (250, 170, 30)),
        Label('traffic sign', 20, 24, 7, 'object', 3, True, False, (220, 220, 0)),
        Label('vegetation', 21, 5, 8, 'nature', 4, False, False, (107, 142, 35)),
        Label('terrain', 22, 4, 9, 'nature', 4, False, False, (152, 251, 152)),
        Label('sky', 23, 9, 10, 'sky', 5, False, False, (70, 130, 180)),
        Label('person', 24, 19, 11, 'human', 6, True, False, (220, 20, 60)),
        Label('rider', 25, 20, 12, 'human', 6, True, False, (255, 0, 0)),
        Label('car', 26, 13, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        Label('truck', 27, 14, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        Label('bus', 28, 34, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        Label('caravan', 29, 16, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        Label('trailer', 30, 15, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        Label('train', 31, 33, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        Label('motorcycle', 32, 17, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        Label('bicycle', 33, 18, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        Label('garage', 34, 12, 2, 'construction', 2, True, False, (64, 128, 128)),
        Label('gate', 35, 6, 4, 'construction', 2, False, False, (190, 153, 153)),
        Label('stop', 36, 29, 255, 'construction', 2, True, True, (150, 120, 90)),
        Label('smallpole', 37, 22, 5, 'object', 3, True, False, (153, 153, 153)),
        Label('lamp', 38, 25, 255, 'object', 3, True, False, (0, 64, 64)),
        Label('trash bin', 39, 26, 255, 'object', 3, True, False, (0, 128, 192)),
        Label('vending machine', 40, 27, 255, 'object', 3, True, False, (128, 64, 0)),
        Label('box', 41, 28, 255, 'object', 3, True, False, (64, 64, 128)),
        Label('unknown construction', 42, 35, 255, 'void', 0, False, True, (102, 0, 0)),
        Label('unknown vehicle', 43, 36, 255, 'void', 0, False, True, (51, 0, 51)),
        Label('unknown object', 44, 37, 255, 'void', 0, False, True, (32, 32, 32)),
        Label('license plate', -1, -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    CLASSES = [label.name for label in LABELS]

    def __init__(
            self,
            config,
            is_train=True,
    ):
        self.cfg = config
        self.is_train = is_train

        if self.is_train:
            self.seqs = self.cfg['DATA']['train_seqs']
        else:
            self.seqs = self.cfg['DATA']['val_seqs']
        masks_dirs = [os.path.join(self.cfg["DATA"]["dataroot"], f'data_3d_semantics/{seq}/bev_maps/') \
                      for seq in self.seqs]
        images_dirs = {
            'image_00': [os.path.join(self.cfg["DATA"]["dataroot"], \
                                      f'data_2d_raw/{seq}/image_00/data_rect/') \
                         for seq in self.seqs],
            'image_01': [os.path.join(self.cfg["DATA"]["dataroot"], \
                                      f'data_2d_raw/{seq}/image_01/data_rect/') \
                         for seq in self.seqs]
        }

        flatten = lambda t: [item for sublist in t for item in sublist]

        self.ids = [os.listdir(masks_dir) for masks_dir in masks_dirs]
        self.cams_images_fps = {
            'image_00': flatten([[os.path.join(imgs_dir, self.maskid_to_imgid(image_id)) \
                                  for image_id in self.ids[idx]] \
                                 for idx, imgs_dir in enumerate(images_dirs['image_00'])]),
            'image_01': flatten([[os.path.join(imgs_dir, self.maskid_to_imgid(image_id)) \
                                  for image_id in self.ids[idx]] \
                                 for idx, imgs_dir in enumerate(images_dirs['image_01'])]),
        }
        self.masks_fps = flatten([[os.path.join(masks_dir, image_id) for image_id in self.ids[idx]] \
                                  for idx, masks_dir in enumerate(masks_dirs)])

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in self.cfg['DATA']['classes']]

        # intrinsics
        calib_dir = '%s/calibration' % (self.cfg["DATA"]["dataroot"])
        self.intrinsic_file = os.path.join(calib_dir, 'perspective.txt')
        self.extrinsics_file = os.path.join(calib_dir, 'calib_cam_to_pose.txt')

        assert os.path.isfile(self.intrinsic_file), '%s does not exist!' % self.intrinsic_file
        self.K, self.width, self.height = self.load_intrinsics()
        # print ('Image size %dx%d ' % (self.height, self.width))
        # print ('Intrinsics \n', self.K)
        assert os.path.isfile(self.extrinsics_file), '%s does not exist!' % self.extrinsics_file
        self.cams_to_pose = self.load_calibration_camera_to_pose()

        print(self)

    def maskid_to_imgid(self, fname):
        base_fname = '0000000000.png'
        img_fname = base_fname[:-len(fname)] + fname
        return img_fname

    def choose_cams(self):
        if self.is_train and self.cfg["DATA"]['ncams'] < len(self.cfg["DATA"]['cams']):
            cams = np.random.choice(self.cfg["DATA"]['cams'], self.cfg["DATA"]['ncams'],
                                    replace=False)
        else:
            cams = self.cfg["DATA"]['cams']
        return cams

    def load_intrinsics(self):
        # load intrinsics
        intrinsic_loaded = False
        width = -1
        height = -1
        with open(self.intrinsic_file) as f:
            intrinsics = f.read().splitlines()
        for line in intrinsics:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                K = [float(x) for x in line[1:]]
                K = np.reshape(K, [3, 4])
                intrinsic_loaded = True
            if line[0] == "S_rect_00:":
                width = int(float(line[1]))
                height = int(float(line[2]))
        assert (intrinsic_loaded == True)
        assert (width > 0 and height > 0)

        return K[:3, :3], width, height

    @staticmethod
    def read_variable(fid, name, M, N):
        # rewind
        fid.seek(0, 0)

        # search for variable identifier
        line = 1
        success = 0
        while line:
            line = fid.readline()
            if line.startswith(name):
                success = 1
                break

        # return if variable identifier not found
        if success == 0:
            return None

        # fill matrix
        line = line.replace('%s:' % name, '')
        line = line.split()
        assert (len(line) == M * N)
        line = [float(x) for x in line]
        mat = np.array(line).reshape(M, N)

        return mat

    def load_calibration_camera_to_pose(self):
        filename = self.extrinsics_file
        with open(filename, 'r') as fid:
            # read variables
            transformation = dict()
            lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
            for camera in self.cfg["DATA"]['cams']:
                transformation[camera] = np.concatenate((self.read_variable(fid, camera, 3, 4), lastrow))
        return transformation

    def sample_augmentation(self):
        H, W = self.cfg["DATA"]['H'], self.cfg["DATA"]['W']
        fH, fW = self.cfg["DATA"]['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.cfg["DATA"]['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.cfg["DATA"]['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.cfg["DATA"]['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.cfg["DATA"]['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.cfg["DATA"]['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, i, cameras):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        for cam in cameras:
            img_path = self.cams_images_fps[cam][i]

            img = Image.open(img_path)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(self.K)
            rot = torch.Tensor(self.cams_to_pose[cam][:3, :3])
            tran = torch.Tensor(self.cams_to_pose[cam][:3, 3])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                       resize=resize,
                                                       resize_dims=resize_dims,
                                                       crop=crop,
                                                       flip=flip,
                                                       rotate=rotate,
                                                       )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_local_map(self, fname):
        mask = cv2.imread(fname, 0)
        # crop mask to fit resolution in configs
        dw = (self.cfg['DATA']['xbound'][1] - self.cfg['DATA']['xbound'][0]) // self.cfg['DATA']['xbound'][2]
        dh = (self.cfg['DATA']['ybound'][1] - self.cfg['DATA']['ybound'][0]) // self.cfg['DATA']['ybound'][2]
        w, h = mask.shape
        mask = mask[:int(dw), int((h - dh) / 2):int((h + dh) / 2)]

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        # mask = cv2.flip(mask, -1) # flip around both axis
        return torch.Tensor(mask.transpose(2, 0, 1))

    def __getitem__(self, i):

        # read data
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(i, cams)
        local_map = self.get_local_map(self.masks_fps[i])

        return imgs, rots, trans, intrins, post_rots, post_trans, local_map

    def __str__(self):
        return f"""KITTI360_Map: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Data Conf: {self.cfg["DATA"]}"""

    def __len__(self):
        return len(self.masks_fps)


class KITTI360_MapOSM(KITTI360_Map):
    def __init__(self, *args, **kwargs):
        super(KITTI360_MapOSM, self).__init__(*args, **kwargs)

        self.osm_path = self.cfg['DATA']['osm_path']
        self.tensor_to_pil = transforms.ToPILImage()

    def __getitem__(self, i):
        # read data
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(i, cams)
        local_map = self.get_local_map(self.masks_fps[i])

        if self.is_train:
            discr_input = self.get_discr_input()
            return imgs, rots, trans, intrins, post_rots, post_trans, local_map, discr_input
        else:
            return imgs, rots, trans, intrins, post_rots, post_trans, local_map

    def get_discr_input(self):
        xmin, xmax, dx = self.cfg['DATA']['xbound']
        ymin, ymax, dy = self.cfg['DATA']['ybound']
        occ_map_size = [int((xmax - xmin) / dx), int((ymax - ymin) / dy)]
        # input for static (road) objects discriminator
        static_discr_input = self.process_discr(self.get_osm(self.osm_path), occ_map_size)
        static_discr_input = torch.transpose(torch.Tensor(static_discr_input), 2, 0)
        # static_discr_input = torch.transpose(static_discr_input, 1, 2)

        # input for dynamic (cars) objects discriminator
        # dynamic_binimg_pil = self.tensor_to_pil(dynamic_binimg)
        # dynamic_discr_input = self.process_discr(dynamic_binimg_pil, self.occ_map_size)
        # dynamic_discr_input = torch.transpose(torch.Tensor(dynamic_discr_input), 2, 0)
        # dynamic_discr_input = torch.transpose(dynamic_discr_input, 1, 2)

        # discr_input = torch.cat([static_discr_input, dynamic_discr_input], dim=0)
        discr_input = static_discr_input
        return discr_input

    @staticmethod
    def get_osm(root_dir):
        # get_osm_path
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)
        with open(osm_path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB').transpose(Image.ROTATE_90)

    def process_discr(self, topview, size):
        topview = self.resize_topview(topview, size)
        topview_n = np.zeros((size[1], size[0], 1))
        topview_n[topview == 255, 0] = 1.
        return topview_n

    @staticmethod
    def resize_topview(topview, size):
        topview = topview.convert("1")
        topview = topview.resize(size, Image.NEAREST)
        topview = topview.convert("L")
        topview = np.array(topview)
        return topview


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name, osm_path='../../monolayout/data/osm/'):
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=dataroot,  # os.path.join(dataroot, version),
                    verbose=False)
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
        'segmentationdatamap': SegmentationDataMap,
        'segmentationdatamaposm': SegmentationDataMapOSM,
    }[parser_name]

    if parser_name == 'segmentationdatamap':
        nusc_maps = get_nusc_maps(dataroot)
        traindata = parser(nusc_maps, nusc, is_train=True, data_aug_conf=data_aug_conf,
                           grid_conf=grid_conf)
        valdata = parser(nusc_maps, nusc, is_train=False, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf)
    elif parser_name == 'segmentationdatamaposm':
        nusc_maps = get_nusc_maps(dataroot)

        valdata = SegmentationDataMapOSM(
                                         osm_path=osm_path,
                                         nusc_maps=nusc_maps,
                                         nusc=nusc,
                                         is_train=False,
                                         )
        traindata = SegmentationDataMapOSM(
                                           osm_path=osm_path,
                                           nusc_maps=nusc_maps,
                                           nusc=nusc,
                                           is_train=True,
                                           )
    else:
        traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                             grid_conf=grid_conf)
        valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                           grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader


