# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')
import math
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from utils.dataset import SyntheticDatasetLDR,RealDatasetLDR
from utils.dataset.scannetpp.dataset import Scannetpp
from utils.path_tracing import ray_intersect
from model.slf import VoxelSLF
from model.brdf import BaseBRDF
from model.emitter import SLFEmitter
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
    parser.add_argument('--slf_path', type=str, required=True)
    parser.add_argument('--emitter_path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True, help='output path')
    parser.add_argument('--dataset', type=str,required=True, help='dataset type')
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


    # create voxle surface light field
    emitter = SLFEmitter(args.emitter_path, args.slf_path)
    for p in emitter.parameters():
        p.requires_grad=False
    emitter.to(device)

    material_net = BaseBRDF()

    denoiser = mitsuba.OptixDenoiser(img_hw[::-1])

    start_time = time.time()

    # bake diffuse shading
    print('bake diffuse')
    output_path = os.path.join(OUTPUT_PATH,'diffuse')
    os.makedirs(output_path,exist_ok=True)
    
    spp = 256
    
    im_id = 0
    for batch in tqdm(dataset):
        rays = batch['rays']
        xs = rays[...,:3]
        ds = rays[...,3:6]

        positions,normals,_,_,valid = ray_intersect(scene,xs.to(device),ds.to(device))
        position = positions[valid]
        normal = normals[valid]
        ds = ds.to(device)[valid]
        
        B = ds.shape[0]
        Ld_ = torch.zeros(B,3,device=device)
        batch_size = 10240*64//spp

        # batched diffuse shading calculation
        for b in range(math.ceil(B*1.0/batch_size)):
            b0 = b*batch_size
            b1 = min(b0+batch_size,B)

            # importance sampling wi
            wi,_,_, = material_net.sample_diffuse(torch.rand((b1-b0)*spp,2,device=device),
                                    normal[b0:b1].repeat_interleave(spp,0))
    
            p_next,_,_,tri_next,valid_next = ray_intersect(scene,
                        position[b0:b1].repeat_interleave(spp,0)+mitsuba.math.RayEpsilon*wi,# prevent self intersection
                        wi.reshape(-1,3))
            
            # query surface light field
            roughness_one = torch.ones_like(tri_next)[:, None]
            Le, _, _ = emitter.eval_emitter(p_next, wi, tri_next, roughness_one, trace_roughness=0.0)
            Ld_[b0:b1] = Le.reshape(b1-b0,spp,3).mean(1)

        # denoiser renderings
        Ld = torch.zeros_like(xs)
        Ld[valid.cpu()] = Ld_.cpu()
        Ld = Ld.reshape(*img_hw,3).numpy()
        Ld = denoiser(Ld).numpy()

        cv2.imwrite(os.path.join(output_path,'{:03d}.exr'.format(im_id)),Ld[:,:,[2,1,0]])
        im_id += 1
    
    print('[bake_shading - diffuse] time (s): ', time.time()-start_time)
    start_time = time.time()

    # bake specular shadings
    print('bake specular')
    output_path = os.path.join(OUTPUT_PATH,'specular')
    os.makedirs(output_path,exist_ok=True)


    spps = [64,128,128,128,128,128] # use different sampling rate
    im_id = 0

    # 6 roughness level
    roughness_level = torch.linspace(0.02,1.0,6)

    for batch in tqdm(dataset):
        rays = batch['rays']
        xs = rays[...,:3]
        ds = rays[...,3:6]
        
        positions,normals,_,_,valid = ray_intersect(scene,xs.to(device),ds.to(device))
        position = positions[valid]
        normal = normals[valid]
        wo = -ds.to(device)[valid]

        B = position.shape[0]
        # caculate for each roughness value
        for r_idx,roughness in enumerate(roughness_level):
            spp = spps[r_idx]
            Ls0_ = torch.zeros(B,3,device=device)
            Ls1_ = torch.zeros(B,3,device=device)
            
            # batched specular shading calculation
            batch_size = 10240*64//spp
            for b in range(math.ceil(B*1.0/batch_size)):
                b0 = b*batch_size
                b1 = min(b0+batch_size,B)
                
                # importance sampling wi
                wi,_,g0,g1 = material_net.sample_specular(
                        torch.rand((b1-b0)*spp,2,device=device),
                        wo[b0:b1].repeat_interleave(spp,0),
                        normal[b0:b1].repeat_interleave(spp,0),roughness
                )

                p_next,_,_, tri_next, valid_next = ray_intersect(scene,
                        position[b0:b1].repeat_interleave(spp,0)+mitsuba.math.RayEpsilon*wi,# prevent self intersection
                        wi.reshape(-1,3))
                
                # query surface light field
                roughness_one = torch.ones_like(tri_next)[:, None]
                Le, _, _ = emitter.eval_emitter(p_next, wi, tri_next, roughness_one, trace_roughness=0.0)

                Ls0_[b0:b1] = (Le*g0).reshape(b1-b0,spp,3).mean(1)
                Ls1_[b0:b1] = (Le*g1).reshape(b1-b0,spp,3).mean(1)

            Ls0 = torch.zeros_like(xs)
            Ls1 = torch.zeros_like(xs)
            Ls0[valid.cpu()] = Ls0_.cpu()
            Ls1[valid.cpu()] = Ls1_.cpu()
            
            Ls0 = Ls0.reshape(*img_hw,3).numpy()
            Ls1 = Ls1.reshape(*img_hw,3).numpy()

            if r_idx > 0: # no need for denoise of low roughness
                Ls0 = denoiser(Ls0).numpy()
                Ls1 = denoiser(Ls1).numpy()
                
            cv2.imwrite(os.path.join(output_path,'{:03d}_0_{}.exr'.format(im_id,r_idx)),Ls0[:,:,[2,1,0]])
            cv2.imwrite(os.path.join(output_path,'{:03d}_1_{}.exr'.format(im_id,r_idx)),Ls1[:,:,[2,1,0]])
        im_id += 1
    
    print('[bake_shading - specular] time (s): ', time.time()-start_time)