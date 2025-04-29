# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as NF
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

from utils.dataset import InvRealDatasetLDR,InvSyntheticDatasetLDR
from utils.dataset.scannetpp.dataset import InvScannetpp
from utils.path_tracing import ray_intersect
import os
import numpy as np
import trimesh
from tqdm import tqdm
from argparse import ArgumentParser
import torch_scatter
from pathlib import Path
from const import set_random_seed
set_random_seed()

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_root', type=str, help='dataset root')
    parser.add_argument('--scene', type=str, required=True, help= 'dataset folder')
    parser.add_argument('--output', type=str, required=True, help='output path')
    parser.add_argument('--dataset', type=str,required=True, help='dataset type')
    parser.add_argument('--mode', type=str, default='export', choices=['export', 'update', 'test'])
    parser.add_argument('--ckpt', type=str,help='checkpoint path')
    parser.add_argument('--spp', type=int,default=100, help='number of samples for each triangle emitter')
    parser.add_argument('--threshold',type=float,default=0.99,help='threshold for emitter')
    parser.add_argument('--ldr_img_dir', type=str, default=None)
    parser.add_argument('--res_scale', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(0)

    SCENE = args.scene
    OUTPUT =args.output
    os.makedirs(OUTPUT, exist_ok=True)

    # load geometry
    if args.dataset in ['synthetic', 'real']:
        mesh_path = os.path.join(args.scene,'scene.obj')
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

    if args.dataset == 'synthetic':
        dataset = InvSyntheticDatasetLDR(SCENE, img_dir=args.ldr_img_dir, split='train', pixel=False)
    elif args.dataset == 'real':
        dataset = InvRealDatasetLDR(SCENE, img_dir=args.ldr_img_dir, split='train', pixel=False)
    elif args.dataset == 'scannetpp':
        dataset = InvScannetpp(args.dataset_root, args.scene, split='train', pixel=False, res_scale=args.res_scale)
    img_hw = dataset.img_hw


    # get mesh vertices and triangles
    if args.mode == 'export':
        mesh = trimesh.load_mesh(mesh_path)
        vertices = torch.from_numpy(np.array(mesh.vertices)).float() #(v, 3)
        faces = torch.from_numpy(np.array(mesh.faces)) #(f, 3)
    
        n_face = len(faces)
        triangle_radiance = torch.zeros(n_face, 3)
        triangle_count = torch.zeros(n_face)
        for batch in tqdm(dataset):
            rays = batch['rays']
            rays_x,rays_d = rays[...,:3].to(device),rays[...,3:6].to(device)
            positions,normals,uvs,triangle_idxs,valid = ray_intersect(scene,rays_x,rays_d)

            triangle_idxs = triangle_idxs[valid].cpu()
            radiance = batch['rgbs'][valid.cpu()]
            # segmentation = batch['segmentation'][valid.cpu()]
            # seg_idxs, inv_idxs = segmentation.unique(return_inverse=True)

            triangle_radiance = torch_scatter.scatter(
                radiance, triangle_idxs, 0, triangle_radiance, reduce='sum'
            )
            triangle_count = torch_scatter.scatter(
                torch.ones(len(triangle_idxs)), triangle_idxs, 0, triangle_count, reduce='sum'
            )

        triangle_radiance_mean = triangle_radiance / triangle_count.unsqueeze(-1).clamp_min(1) #(f, 3)
        triangle_radiance_mean = torch.max(triangle_radiance_mean, dim=-1)[0] # max value among 3 channels

        is_emitter = triangle_radiance_mean > args.threshold
        emitter_vertices = vertices[faces[is_emitter]]
        emitter_area = torch.cross(emitter_vertices[:,1]-emitter_vertices[:,0],
                                   emitter_vertices[:,2]-emitter_vertices[:,0],-1)
        emitter_normal = NF.normalize(emitter_area,dim=-1)
        emitter_area = emitter_area.norm(dim=-1)/2.0
        emitter_radiance = torch.zeros(n_face, 3)

        # print('is_emitter ratio:', torch.sum(is_emitter) / is_emitter.numel())
        torch.save({
            'is_emitter': is_emitter,
            'emitter_vertices': emitter_vertices,
            'emitter_area': emitter_area,
            'emitter_normal': emitter_normal,
            'emitter_radiance': emitter_radiance
        }, os.path.join(OUTPUT,'emitter.pth'))

    if args.mode == 'update':
        ckpt_state = torch.load(args.ckpt, map_location='cpu')['state_dict']
        emitter_path = os.path.join(OUTPUT, 'emitter.pth')
        emitter_state = torch.load(emitter_path, map_location='cpu')
        emitter_state['emitter_radiance'] = ckpt_state['emitter.radiance']
        torch.save(emitter_state, emitter_path)

if __name__ == '__main__':
    main()