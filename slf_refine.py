# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from utils.dataset import SyntheticDatasetLDR,RealDatasetLDR
from utils.dataset.scannetpp.dataset import Scannetpp
from utils.path_tracing import ray_intersect
from model.slf import VoxelSLF
from model.brdf import BaseBRDF
from crf.model_crf import EmorCRF
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from const import set_random_seed
set_random_seed()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_root', type=str, help='dataset root')
    parser.add_argument('--scene', type=str, required=True, help='dataset folder')
    parser.add_argument('--output', type=str, required=True, help='output path')
    parser.add_argument('--load', type=str, default='vslf.npz')
    parser.add_argument('--save', type=str, default='vslf.npz')
    parser.add_argument('--dataset', type=str,required=True, help='dataset type')
    parser.add_argument('--voxel_num', type=int,default=256, help='resolution for voxel radiance cache')
    parser.add_argument('--ldr_img_dir', type=str, default=None)
    parser.add_argument('--ckpt', type=str, help='checkpoint to load CRF model')
    parser.add_argument('--crf_basis', type=int, help='number of CRF basis')
    parser.add_argument('--res_scale', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(0) # use gpu device 0

    DATASET_PATH = args.scene
    OUTPUT_PATH = args.output
    os.makedirs(OUTPUT_PATH,exist_ok=True)
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
    if args.dataset  == 'synthetic':
        dataset = SyntheticDatasetLDR(DATASET_PATH, img_dir=args.ldr_img_dir, split='train', pixel=False)
    elif args.dataset == 'real':
        dataset = RealDatasetLDR(DATASET_PATH, img_dir=args.ldr_img_dir, split='train', pixel=False)
    elif args.dataset == 'scannetpp':
        dataset = Scannetpp(args.dataset_root, args.scene, split='train', pixel=False, res_scale=args.res_scale)
    img_hw = dataset.img_hw
    
    model_crf = EmorCRF(args.crf_basis)
    if args.ckpt:
        state_dict = torch.load(args.ckpt, map_location='cpu')['state_dict']
        weight = {}
        for k,v in state_dict.items():
            if 'model_crf.' in k:
                weight[k.replace('model_crf.','')]=v
        model_crf.load_state_dict(weight)
    model_crf = model_crf.to(device)

    vslf_load = os.path.join(args.output, args.load)
    vslf_save = os.path.join(args.output, args.save)

    with torch.no_grad():
        device = model_crf.weight.device
        state_dict = torch.load(vslf_load,map_location='cpu') 
        vslf = VoxelSLF(state_dict['mask'], state_dict['voxel_min'],state_dict['voxel_max'])
        for idx in tqdm(range(len(dataset)), postfix='Update VSLF'):
            batch = dataset[idx]
            rays = batch['rays'].to(device)
            radiance = batch['rgbs'].to(device)
            xs = rays[...,:3]
            ds = rays[...,3:6]

            exposure = batch['exposure']
            radiance = model_crf.inverse(radiance, exposure)

            positions,_,_,_,valid = ray_intersect(scene,xs,ds)
            if not valid.any():
                continue

            vslf.scatter_add(positions[valid].cpu(),radiance.to(device)[valid].cpu())

        # average pooling the radiance
        vslf.radiance = vslf.radiance/vslf.count[...,None].float().clamp_min(1)
        state_dict['weight'] = vslf.state_dict()
        torch.save(state_dict, vslf_save)