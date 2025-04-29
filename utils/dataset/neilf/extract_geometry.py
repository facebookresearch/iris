# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

import mitsuba
mitsuba.set_variant('cuda_ad_rgb')
import numpy as np
import torch 
import torch.nn.functional as F
import math

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from utils.dataset import SyntheticDataset,RealDataset,SyntheticDatasetLDR,RealDatasetLDR
from utils.dataset.scannetpp.dataset import Scannetpp
from utils.path_tracing import ray_intersect
from model.brdf import BaseBRDF
from model.emitter import SLFEmitter
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import time

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_root', type=str, help='dataset root')
    parser.add_argument('--scene', type=str, required=True, help='dataset folder')
    parser.add_argument('--output', type=str, required=True, help='output path')
    parser.add_argument('--dataset', type=str,required=True, help='dataset type')
    parser.add_argument('--input', type=str, default='ldr', choices=['hdr', 'ldr'])
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--ldr_img_dir', type=str, default=None)
    parser.add_argument('--res_scale', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(0) # use gpu device 0

    DATASET_PATH = args.scene
    OUTPUT_PATH = args.output
    os.makedirs(OUTPUT_PATH,exist_ok=True)
    dir_position = os.path.join(OUTPUT_PATH, 'position_maps')
    dir_normal   = os.path.join(OUTPUT_PATH, 'normal_maps')
    dir_image    = os.path.join(OUTPUT_PATH, 'images')
    dir_depth    = os.path.join(OUTPUT_PATH, 'depth_maps')
    os.makedirs(dir_position, exist_ok=True)
    os.makedirs(dir_normal, exist_ok=True)
    os.makedirs(dir_image, exist_ok=True)
    os.makedirs(dir_depth, exist_ok=True)

    # load mesh
    if args.dataset in ['synthetic', 'real']:
        mesh_path = os.path.join(DATASET_PATH,'scene.obj')
        mesh_type = 'obj'
    elif args.dataset == 'scannetpp':
        mesh_path = os.path.join(args.dataset_root, 'data', args.scene, 'scans', 'scene.ply')
        mesh_type = 'ply'
    assert Path(mesh_path).exists(), 'mesh not found: '+mesh_path
    
    scene = mitsuba.load_dict({
        'type': 'scene',
        'shape_id':{
            'type': mesh_type,
            'filename': mesh_path, 
        }
    })

    # load dataset
    if args.input == 'hdr':
        if args.dataset  == 'synthetic':
            dataset = SyntheticDataset(DATASET_PATH, split=args.split, pixel=False)
        elif args.dataset == 'real':
            dataset = RealDataset(DATASET_PATH, split=args.split, pixel=False)
    else:
        if args.dataset  == 'synthetic':
            dataset = SyntheticDatasetLDR(DATASET_PATH, img_dir=args.ldr_img_dir, split=args.split, pixel=False)
        elif args.dataset == 'real':
            dataset = RealDatasetLDR(DATASET_PATH, img_dir=args.ldr_img_dir, split=args.split, pixel=False)
        elif args.dataset == 'scannetpp':
            dataset = Scannetpp(args.dataset_root, args.scene, split=args.split, pixel=False, res_scale=args.res_scale)
    img_hw = dataset.img_hw
    img_hw = dataset.img_hw


    denoiser = mitsuba.OptixDenoiser(img_hw[::-1])
    
    im_id = 0
    for i in tqdm(range(len(dataset))):
        name = dataset.img_name_list[i].split('.')[0]
        batch = dataset[i]
        rays = batch['rays']
        xs = rays[...,:3]
        ds = rays[...,3:6]
        rgb = batch['rgbs']
        rgb = rgb.reshape(*img_hw, 3).cpu().numpy()
        rgb = (rgb*255).astype(np.uint8)
        rgb_path = os.path.join(dir_image, '{}.png'.format(name))
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        positions,normals,_,_,valid = ray_intersect(scene,xs.to(device),ds.to(device))
        positions = positions.reshape(*img_hw, 3).cpu().numpy()
        normals = normals.reshape(*img_hw, 3).cpu().numpy()
        ds = F.normalize(ds, dim=-1).reshape(*img_hw, 3).cpu().numpy()
        xs = xs.reshape(*img_hw, 3).cpu().numpy()
        depths = np.sum((positions - xs) * ds, axis=-1)

        position_path = os.path.join(dir_position, '{}.exr'.format(name))
        cv2.imwrite(position_path, positions)
        normal_path = os.path.join(dir_normal, '{}.exr'.format(name))
        cv2.imwrite(normal_path, normals)
        depth_path = os.path.join(dir_depth, '{}.npy'.format(name))
        np.save(depth_path, depths)