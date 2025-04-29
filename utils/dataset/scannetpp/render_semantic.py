# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import cv2 
import torch 
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')
import trimesh
from .dataset_dslr import Scannetpp
from utils.path_tracing import ray_intersect
from tqdm import tqdm
import argparse 
from const import set_random_seed
set_random_seed()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/hdd/datasets/scannetpp/')
    parser.add_argument('--scene')
    args = parser.parse_args()
    dataset = Scannetpp(args.data_root, args.scene, split='all', pixel=False)

    mesh_path = os.path.join(args.data_root, 'data', args.scene, 'scans', 'mesh.ply')
    mesh_type = 'ply'
    scene = mitsuba.load_dict({
        'type': 'scene',
        'shape_id':{
            'type': mesh_type,
            'filename': mesh_path, 
        }
    })

    sem_mesh_path = os.path.join(args.data_root, 'data', args.scene, 'scans', 'mesh_aligned_0.05_semantic.ply')
    sem_mesh = trimesh.load(sem_mesh_path)
    vertices = np.array(sem_mesh.vertices) # (v, 3)
    faces    = np.array(sem_mesh.faces)    # (f, 3)
    
    vtx_colors = (sem_mesh.visual.vertex_colors).astype(np.int64)
    vtx_values = vtx_colors[:, 0] + vtx_colors[:, 1]*(2**8) + vtx_colors[:, 2]*(2**16)
    seg_idxs, inv_idxs = np.unique(vtx_values, return_inverse=True)
    # face_colors = (sem_mesh.visual.face_colors).astype(np.int64)
    # face_values = face_colors[:, 0] + face_colors[:, 1]*(2**8) + face_colors[:, 2]*(2**16)
    # seg_idxs, inv_idxs = np.unique(face_values, return_inverse=True)
    seg_idxs_new = np.arange(len(seg_idxs))
    vtx_labels = seg_idxs_new[inv_idxs]


    img_name_list = dataset.img_name_list
    dir_sem_out = os.path.join(args.data_root, 'data', args.scene, 'dslr', 'seg')
    os.makedirs(dir_sem_out, exist_ok=True)

    h, w = dataset.img_hw
    device = torch.device(0)
    colors = np.random.rand(len(seg_idxs), 3)
    for i in tqdm(range(len(dataset))):
        name = img_name_list[i]
        batch = dataset[i]
        rays = batch['rays']
        rays_x,rays_d = rays[...,:3].to(device),rays[...,3:6].to(device)
        _, _, _, face_idxs, valid = ray_intersect(scene,rays_x,rays_d)
        
        face_idxs = face_idxs.cpu().numpy()
        valid = valid.cpu().numpy()
        valid_face_idxs = face_idxs[valid]   # (n, )
        valid_faces = faces[valid_face_idxs] # (n, 3)
        
        valid_labels = vtx_labels[valid_faces] # (n, 3)
        valid_labels = np.max(valid_labels, axis=1) #choose max id as label
        labels = np.zeros((h*w))
        labels[~valid] = -1
        labels[valid] = valid_labels
        labels = labels.reshape(h, w).astype(np.uint8)
        save_path = os.path.join(dir_sem_out, name)
        cv2.imwrite(save_path, labels)

        labels_vis = np.zeros((h*w, 3))
        labels_vis[valid] = colors[valid_labels]
        labels_vis = (labels_vis*255).reshape(h, w, -1).astype(np.uint8)
        save_path = os.path.join(dir_sem_out, name.split('.')[0] + '_vis.jpg')
        cv2.imwrite(save_path, labels_vis)

if __name__ == '__main__':
    main()