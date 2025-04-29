# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

from utils.dataset import SyntheticDatasetLDR,RealDatasetLDR
from utils.dataset.scannetpp.dataset import Scannetpp
from utils.path_tracing import ray_intersect
from model.slf import VoxelSLF
from crf.model_crf import EmorCRF
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
import time
from const import set_random_seed
set_random_seed()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_root', type=str, help='dataset root')
    parser.add_argument('--scene', type=str, required=True, help='dataset folder')
    parser.add_argument('--output', type=str, required=True, help='output path')
    parser.add_argument('--dataset', type=str,required=True, help='dataset type')
    parser.add_argument('--voxel_num', type=int,default=256, help='resolution for voxel radiance cache')
    parser.add_argument('--ldr_img_dir', type=str, default=None)
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
    
    model_crf = EmorCRF()

    start_time = time.time()
    # extract scene bounding box
    print('find scene bound')
    voxel_min = 1000.
    voxel_max = 0.0
    for idx in tqdm(range(len(dataset))):
        batch = dataset[idx]
        rays = batch['rays']
        xs = rays[...,:3]
        ds = rays[...,3:6]

        positions,_,_,_,valid = ray_intersect(scene,xs.to(device),ds.to(device))
        if not valid.any():
            continue
        position = positions[valid]
        voxel_min = min(voxel_min,position.min())
        voxel_max = max(voxel_max,position.max())
    
    if args.dataset in ['synthetic', 'real']:
        voxel_min = 1.1*voxel_min
        voxel_max = 1.1*voxel_max
    else:
        voxel_c = (voxel_min + voxel_max)
        voxel_min = voxel_c + (voxel_min - voxel_c) * 1.1
        voxel_max = voxel_c + (voxel_max - voxel_c) * 1.1

    # find voxels that are not occupied
    print('find visible voxels')
    res_spatial = args.voxel_num
    SpatialHist = torch.zeros(res_spatial**3,device=device)
    for idx in tqdm(range(len(dataset))):
        batch = dataset[idx]
        rays = batch['rays']
        xs = rays[...,:3]
        ds = rays[...,3:6]

        positions,_,_,_,valid = ray_intersect(scene,xs.to(device),ds.to(device))
        if not valid.any():
            continue
        
        position = (positions[valid]-voxel_min)/(voxel_max-voxel_min)
        position = (position*res_spatial).long().clamp(0,res_spatial-1)
        inds = position[...,0] + position[...,1]*res_spatial\
             + position[...,2]*res_spatial*res_spatial
        SpatialHist.scatter_add_(0,inds,torch.ones_like(inds).float())
    SpatialHist = SpatialHist.reshape(res_spatial,res_spatial,res_spatial)

    mask = (SpatialHist>0)
    
    # create voxle surface light field
    print('bake voxel surface light field')
    vslf = VoxelSLF(mask.cpu(),voxel_min.item(),voxel_max.item())
    for idx in tqdm(range(len(dataset))):
        batch = dataset[idx]
        rays = batch['rays']
        radiance = batch['rgbs']
        xs = rays[...,:3]
        ds = rays[...,3:6]

        exposure = batch['exposure']
        radiance = model_crf.inverse(radiance, exposure)

        positions,_,_,_,valid = ray_intersect(scene,xs.to(device),ds.to(device))
        if not valid.any():
            continue

        vslf.scatter_add(positions[valid].cpu(),radiance.to(device)[valid].cpu())

    # average pooling the radiance
    vslf.radiance = vslf.radiance/vslf.count[...,None].float().clamp_min(1)
    
    torch.save({
        'mask': (SpatialHist>0),
        'voxel_min': voxel_min.item(),
        'voxel_max': voxel_max.item(),
        'weight':vslf.state_dict()
    },os.path.join(OUTPUT_PATH,'vslf.npz'))
