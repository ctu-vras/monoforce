#!/usr/bin/env python

import sys
sys.path.append('../src')
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from monoforce.models.terrain_encoder.utils import denormalize_img, ego_to_cam, get_only_in_img_mask
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from eval import Evaluator
from monoforce.datasets.rough import ROUGH
from monoforce.utils import read_yaml, write_to_yaml, str2bool, compile_data
from monoforce.losses import hm_loss, physics_loss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description='Train MonoForce model')
    parser.add_argument('--model', type=str, default='lss', help='Model to train: lss')
    parser.add_argument('--bsz', type=int, default=4, help='Batch size')
    parser.add_argument('--nepochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--robot', type=str, default='marv', help='Robot name')
    parser.add_argument('--lss_cfg_path', type=str, default='../config/lss_cfg.yaml', help='Path to LSS config')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--debug', type=str2bool, default=True, help='Debug mode: use small datasets')
    parser.add_argument('--vis', type=str2bool, default=False, help='Visualize training samples')
    parser.add_argument('--geom_weight', type=float, default=1.0, help='Weight for geometry loss')
    parser.add_argument('--terrain_weight', type=float, default=2.0, help='Weight for terrain heightmap loss')
    parser.add_argument('--phys_weight', type=float, default=1.0, help='Weight for physics loss')
    parser.add_argument('--traj_sim_time', type=float, default=5.0, help='Trajectory simulation time')
    parser.add_argument('--dphys_grid_res', type=float, default=0.4, help='DPhys grid resolution')

    return parser.parse_args()


class Trainer(Evaluator):
    pass


def main():
    args = arg_parser()
    print(args)


if __name__ == '__main__':
    main()