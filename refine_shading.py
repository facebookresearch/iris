# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')
import math

from utils.dataset import RealDatasetLDR,SyntheticDatasetLDR
from utils.dataset.scannetpp.dataset import Scannetpp
from utils.ops import *
from utils.path_tracing import *
from model.emitter import SLFEmitter
from model.brdf import NGPBRDF
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
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
    parser.add_argument('--slf_path', type=str, required=True)
    parser.add_argument('--emitter_path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True, help='last shading folder')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint path')
    parser.add_argument('--dataset',type=str,required=True, help='dataset type')
    parser.add_argument('--ldr_img_dir', type=str, default=None)
    parser.add_argument('--res_scale', type=float, default=1.0)
    args = parser.parse_args()
    device = torch.device(0)

    DATASET_PATH = args.scene
    OUTPUT_PATH = args.output
    os.makedirs(OUTPUT_PATH,exist_ok=True)

    # load geometry
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

    # load emitter
    emitter = SLFEmitter(args.emitter_path, args.slf_path)
    for p in emitter.parameters():
        p.requires_grad=False
    emitter.to(device)
    for p in emitter.parameters():
        p.requires_grad=False

    # load brdf
    mask = torch.load(args.slf_path, map_location='cpu')
    material_net = NGPBRDF(mask['voxel_min'],mask['voxel_max'])
    state_dict = torch.load(args.ckpt,map_location='cpu')['state_dict']
    weight = {}
    for k,v in state_dict.items():
        if 'material.' in k:
            weight[k.replace('material.','')] = v
    material_net.load_state_dict(weight)
    material_net.to(device)
    for p in material_net.parameters():
        p.requires_grad=False

    # set up denoiser
    denoiser = mitsuba.OptixDenoiser(img_hw[::-1])
    
    start_time = time.time()
    
    # refine diffuse shading
    print('refine diffuse')
    output_path = os.path.join(OUTPUT_PATH,'diffuse')
    os.makedirs(output_path,exist_ok=True)
    spp = 128
    indir_depth = 5
    
    # batched process
    batch_size = 10240*128//spp
    im_id = 0
    for batch in tqdm(dataset):
        rays = batch['rays']
        rays_x,rays_d = rays[...,:3].to(device),rays[...,3:6].to(device)
        positions,normals,uvs,triangle_idxs,valid = ray_intersect(scene,rays_x,rays_d)
        wi = rays_d
        B = len(positions)
        L = torch.zeros(B,3,device=device)
        for b in range(math.ceil(B*1.0/batch_size)):
            b0 = b*batch_size
            b1 = min(b0+batch_size,B)
            L[b0:b1] = path_tracing_det_diff(scene,emitter,material_net,
                                             positions[b0:b1],wi[b0:b1],normals[b0:b1],
                                             uvs[b0:b1],triangle_idxs[b0:b1],
                                             spp,indir_depth)
        assert L.isnan().any() == False
        L = denoiser(mitsuba.TensorXf(L.reshape(*img_hw,3))).numpy()
        cv2.imwrite(os.path.join(output_path,'{:03d}.exr'.format(im_id)),L[:,:,[2,1,0]])
        im_id += 1


    print('[refine_shading - diffuse] time (s): ', time.time()-start_time)
    start_time = time.time()

    # refine spacular shadings
    print('refine specular')
    output_path = os.path.join(OUTPUT_PATH,'specular')
    os.makedirs(output_path,exist_ok=True)
    spp = 64

    batch_size = 10240*128//spp
    im_id = 0

    # 6 roughness level
    roughness_level = torch.linspace(0.02,1.0,6)
    for batch in tqdm(dataset):
        rays = batch['rays']
        rays_x,rays_d = rays[...,:3].to(device),rays[...,3:6].to(device)
        positions,normals,uvs,triangle_idxs,valid = ray_intersect(scene,rays_x,rays_d)
        wi = rays_d
        B = len(positions)
        L0 = torch.zeros(B,3,device=device)
        L1 = L0.clone()

        for r_idx,roughness in enumerate(roughness_level):
            # BxSx3

            B = len(positions)
            L0 = torch.zeros(B,3,device=device)
            L1 = L0.clone()

            for b in range(math.ceil(B*1.0/batch_size)):
                b0 = b*batch_size
                b1 = min(b0+batch_size,B)
                L0_,L1_ = path_tracing_det_spec(scene,emitter,material_net,
                                                 roughness,
                                                 positions[b0:b1],wi[b0:b1],normals[b0:b1],
                                                 uvs[b0:b1],triangle_idxs[b0:b1],
                                                 spp,indir_depth)
                L0[b0:b1] = L0_
                L1[b0:b1] = L1_
            assert L0.isnan().any() == False
            assert L1.isnan().any() == False
            L0 = denoiser(mitsuba.TensorXf(L0.reshape(*img_hw,3))).numpy()
            L1 = denoiser(mitsuba.TensorXf(L1.reshape(*img_hw,3))).numpy()
            cv2.imwrite(os.path.join(output_path,'{:03d}_0_{}.exr'.format(im_id,r_idx)),L0[:,:,[2,1,0]])
            cv2.imwrite(os.path.join(output_path,'{:03d}_1_{}.exr'.format(im_id,r_idx)),L1[:,:,[2,1,0]])
        im_id += 1

    print('[refine_shading - specular] time (s): ', time.time()-start_time)
