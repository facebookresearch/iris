# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as NF
from torch.utils.data import Dataset
import json
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import math
from pathlib import Path
from .synthetic_ldr import open_png
from PIL import Image
from const import GAMMA, set_random_seed
set_random_seed()

def normalize_v(x) -> np.ndarray:
    return x / np.linalg.norm(x)

def read_cam_params(camFile: Path) -> list:
    """ read open gl camera """
    assert camFile.exists()
    with open(str(camFile), 'r') as camIn:
        cam_data = camIn.read().splitlines()
    cam_num = int(cam_data[0])
    cam_params = np.array([x.split(' ') for x in cam_data[1:]]).astype(np.float32)
    assert cam_params.shape[0] == cam_num * 3
    cam_params = np.split(cam_params, cam_num, axis=0) # [[origin, lookat, up], ...]
    return cam_params

def open_exr(file,img_hw):
    """ open image exr file """
    img = cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = img[...,[2,1,0]]
    hs, ws, _ = img.shape
    ht, wt    = img_hw
    if (ht != hs) or (wt != ws):
        img = cv2.resize(img, (wt, ht))
    img = torch.from_numpy(img.astype(np.float32))
    return img


def get_direction(k,img_hw):
    """ get camera ray direction (unormzlied)
        k: 3x3 camera intrinsics
        img_hw: image height and width
    """
    screen_y,screen_x = torch.meshgrid(torch.linspace(0.5,img_hw[0]-0.5,img_hw[0]),
                                torch.linspace(0.5,img_hw[1]-0.5,img_hw[1]))
    rays_d = torch.stack([
        (screen_x-k[0,2])/k[0,0],
        (screen_y-k[1,2])/k[1,1],
        torch.ones_like(screen_y)
    ],-1).reshape(-1,3)
    return rays_d

def to_world(rays_d,c2w,ray_diff,k):
    """ world sapce camera ray origin and direction
    Args:
        rays_d: HWx3 unormalized camera ray direction (local)
        c2w: 3x4 camera to world matrix
        ray_diff: True if return ray differentials
        k: 3x3 camera intrinsics
    Return:
        HWx3 camera origin
        HWx3 camera direction (unormzlied) if ray_diff==True
        HWx3 dxdu if ray_diff==True
        HWx3 dydv if ray_diff==True
    """
    rays_x = c2w[:,3:].T*torch.ones_like(rays_d)
    rays_d = rays_d@c2w[:3,:3].T
    if ray_diff:
        dxdu = torch.tensor([1.0/k[0,0],0,0])[None].expand_as(rays_d)@c2w[:3,:3].T
        dydv = torch.tensor([0,1.0/k[1,1],0])[None].expand_as(rays_d)@c2w[:3,:3].T
        return rays_x,rays_d,dxdu,dydv
    else:
        return rays_x,NF.normalize(rays_d,dim=-1)

def get_split_ids(n_total, split='train'):
    val_ids = [i*10 for i in range(16)]
    train_ids = [i for i in range(n_total) if i not in val_ids]
    if split == 'train':
        return train_ids
    else:
        return val_ids 

class RealDatasetLDR(Dataset):
    """ Real world capture dataset in structure:
    Scene/
        Image/{:03d}_0001.exr HDR images
        segmentation/{:03d}.exr semantic segmentations
        cam.txt Image extrinsics
        K_list.txt Image intrinsics
        scene.obj Scene mesh
    """
    def __init__(self, 
                 root_dir, 
                 img_dir=None, 
                 split='train', 
                 pixel=True,
                 ray_diff=False, 
                 load_traj=False, 
                 val_frame=0,
                 res_scale=1.0):
        """
        Args:
            root_dir: dataset root folder
            split: train or val
            pixel: whether load every camera pixel
            ray_diff: whether return ray differentials
        """
        self.root_dir = root_dir
        self.pixel=pixel
        self.split = split
        if img_dir == None:
            self.img_dir = 'Image'
            self.exposures = None 
            self.crfs      = None
            self.multi_exposure = False
            self.gamma = GAMMA
        else:
            self.img_dir = img_dir 
            self.exposures = np.load(os.path.join(self.root_dir, self.img_dir, 'cam', 'exposure.npy'))
            self.crfs      = np.load(os.path.join(self.root_dir, self.img_dir, 'cam', 'crf.npy')) #(3, 1024)
            self.multi_exposure = True
            self.gamma = None

        # find image hight x width
        h, w = cv2.imread(os.path.join(root_dir,'Image/000_0001.exr'),-1).shape[:2]
        self.img_hw = (int(h*res_scale), int(w*res_scale))
        self.ray_diff = ray_diff
        self.val_frame = val_frame

        C2Ws_raw = read_cam_params(Path(self.root_dir,'cam.txt'))
        C2Ws = []
        # convert to opencv
        for i,c2w_raw in enumerate(C2Ws_raw):
            origin, lookat, up = np.split(c2w_raw.T, 3, axis=1)
            origin = origin.flatten()
            lookat = lookat.flatten()
            up = up.flatten()
            at_vector = normalize_v(lookat - origin)
            assert np.amax(np.abs(np.dot(at_vector.flatten(), up.flatten()))) < 2e-3 # two vector should be perpendicular

            t = origin.reshape((3, 1)).astype(np.float32)
            R = np.stack((np.cross(-up, at_vector), -up, at_vector), -1).astype(np.float32)
            C2Ws.append(np.hstack((R, t)))
        Ks = read_cam_params(Path(self.root_dir,'K_list.txt'))
        
        C2Ws = np.stack(C2Ws,0)
        Ks = np.stack(Ks,0)
        Ks[:, :2, :] *= res_scale
        # split into train/val
        split_ids = get_split_ids(C2Ws.shape[0], split)
        self.split_ids = split_ids
        C2Ws = C2Ws[split_ids]
        Ks = Ks[split_ids]
        if self.multi_exposure:
            self.exposures = self.exposures[split_ids]

        C2Ws = torch.from_numpy(C2Ws).float()
        Ks = torch.from_numpy(Ks).float()
        
        self.C2Ws = C2Ws
        self.Ks = Ks
        
        if self.pixel:
            # load all camera pixels
            self.all_rays = []
            for idx in range(len(self.Ks)):
                k = self.Ks[idx]
                c2w = self.C2Ws[idx]
                img_hw = self.img_hw
                img_idx = self.split_ids[idx]
                img = open_png(Path(self.root_dir,     
                    self.img_dir,'{:03d}_0001.png'.format(img_idx)),img_hw,self.gamma).reshape(-1,3).clamp_min(0)
                
                rays_d = get_direction(k,img_hw)
                
                if self.ray_diff:
                    # load ray differential
                    rays_x,rays_d,dxdu,dydv = to_world(rays_d,c2w,self.ray_diff,k)
                    self.all_rays.append(torch.cat([
                        rays_x,rays_d,dxdu,dydv,
                        img
                    ],-1))
                else:
                    rays_x,rays_d = to_world(rays_d,c2w,self.ray_diff,k)
                    self.all_rays.append(torch.cat([
                        rays_x,rays_d,
                        img
                    ],-1))
            self.all_rays = torch.cat(self.all_rays,0)
            if self.multi_exposure:
                pixel_num = self.img_hw[0] * self.img_hw[1]
                self.exposures = torch.tensor(self.exposures).view(-1, 1, 1).repeat(1, pixel_num, 1).view(-1, 1)
        
        if load_traj:
            render_traj = np.load(os.path.join(root_dir, 'render_traj.npy'))
            self.render_traj_c2w = render_traj
            k = self.Ks[0]
            rays_d_cam = get_direction(k, self.img_hw)
            self.render_traj_rays = []
            for i in range(len(render_traj)):
                c2w = torch.FloatTensor(render_traj[i])
                rays_o,rays_d,dxdu,dydv = to_world(rays_d_cam,c2w,self.ray_diff,k)
                self.render_traj_rays += [torch.cat([rays_o, rays_d, dxdu, dydv],1)]
    
    def __len__(self,):
        if self.pixel == True:
            return len(self.all_rays)
        # if self.split == 'val':
        #     # only load 8 images for validation of reconstruction
        #     return 8
        return len(self.C2Ws)
    
    def __getitem__(self,idx):
        exposure = None
        if self.pixel:
            if self.multi_exposure:
                exposure = self.exposures[idx]
            tmp = self.all_rays[idx]
            return {
                'rays': tmp[...,:-3],
                'rgbs': tmp[...,-3:],
                'exposure': exposure
            }
        k = self.Ks[idx]
        c2w = self.C2Ws[idx]
        img_hw = self.img_hw

        img_idx = self.split_ids[idx]
        img = open_png(Path(self.root_dir,     
            self.img_dir,'{:03d}_0001.png'.format(img_idx)),img_hw,self.gamma).reshape(-1,3).clamp_min(0)
        rays_d = get_direction(k,img_hw)
        
        if self.ray_diff:
            rays_x,rays_d,dxdu,dydv = to_world(rays_d,c2w,self.ray_diff,k)
            rays = torch.cat([rays_x,rays_d,dxdu,dydv],-1)
        else:
            rays_x,rays_d = to_world(rays_d,c2w,self.ray_diff,k)
            rays = torch.cat([rays_x,rays_d,],-1)
        if self.multi_exposure:
            exposure = self.exposures[idx]
        return {
            'rays': rays,
            'rgbs': img,
            'c2w': c2w,
            'img_hw': img_hw,
            'exposure': exposure
        }
    
    
    
class InvRealDatasetLDR(Dataset):
    """ Real world capture dataset with diffuse and specular shadings
    Shading folder in structure:
    Scene/
        diffuse/{:03d}.exr diffuse shadings (L_d)
        specular/{:03d}_i_j.exr specular shadings (L_s^i(\sigma_j))
    """
    def __init__(self, root_dir, img_dir=None, batch_size=None,split='train', pixel=True, val_frame=0, cache_dir=None):
        """
        Args:
            root_dir: dataset root folder
            cache_dir: shadings folder
            split: train or val
            pixel: whether load every camera pixel
            batch_size: size of each ray batch if pixel==True
        """
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        if img_dir == None:
            self.img_dir = 'Image'
            self.albedo_dir = 'irisformer/albedo'
            self.exposures  = None 
            self.crfs       = None
            self.multi_exposure = False
            self.gamma = GAMMA
        else:
            self.img_dir = img_dir 
            self.albedo_dir = os.path.join(img_dir, 'albedo')
            self.exposures  = np.load(os.path.join(self.root_dir, self.img_dir, 'cam', 'exposure.npy'))
            self.crfs       = np.load(os.path.join(self.root_dir, self.img_dir, 'cam', 'crf.npy')) #(3, 1024)
            self.multi_exposure = True
            self.gamma = None
        self.pixel=pixel
        self.split = split
        self.ray_diff = True
        self.val_frame = val_frame

        self.img_hw = cv2.imread(os.path.join(root_dir,'Image/000_0001.exr'),-1).shape[:2]
        self.batch_size = batch_size
        # approximate roughness channel by interpolating 6 samples
        self.roughness_level = 6
        
        C2Ws_raw = read_cam_params(Path(self.root_dir,'cam.txt'))
        C2Ws = []
        for i,c2w_raw in enumerate(C2Ws_raw):
            origin, lookat, up = np.split(c2w_raw.T, 3, axis=1)
            origin = origin.flatten()
            lookat = lookat.flatten()
            up = up.flatten()
            at_vector = normalize_v(lookat - origin)
            assert np.amax(np.abs(np.dot(at_vector.flatten(), up.flatten()))) < 2e-3 # two vector should be perpendicular

            t = origin.reshape((3, 1)).astype(np.float32)
            R = np.stack((np.cross(-up, at_vector), -up, at_vector), -1).astype(np.float32)
            C2Ws.append(np.hstack((R, t)))
        Ks = read_cam_params(Path(self.root_dir,'K_list.txt'))
        
        C2Ws = np.stack(C2Ws,0)
        Ks = np.stack(Ks,0)
        # split into train/val
        split_ids = get_split_ids(C2Ws.shape[0], split)
        self.split_ids = split_ids
        C2Ws = C2Ws[split_ids]
        Ks = Ks[split_ids]
        if self.multi_exposure:
            self.exposures = self.exposures[split_ids]

        C2Ws = torch.from_numpy(C2Ws).float()
        Ks = torch.from_numpy(Ks).float()
        
        self.C2Ws = C2Ws
        self.Ks = Ks
        
        if self.pixel:
            self.all_rays = []
            self.all_intrinsic = []
            self.all_cache = []
            for idx in range(len(self.Ks)):
                k = self.Ks[idx]
                c2w = self.C2Ws[idx]
                img_hw = self.img_hw
                img_idx = self.split_ids[idx]
                img = open_png(Path(self.root_dir,     
                    self.img_dir,'{:03d}_0001.png'.format(img_idx)),img_hw).reshape(-1,3).clamp_min(0)

                # segmentation mask
                segmentation = torch.from_numpy(cv2.imread(os.path.join(self.root_dir,'segmentation','{:03d}.exr'.format(img_idx)),-1))[...,0].reshape(-1,1).float()
                
                rays_d = get_direction(k,img_hw)
                rays_x,rays_d,dxdu,dydv = to_world(rays_d,c2w,self.ray_diff,k)
                self.all_rays.append(torch.cat([
                    rays_x, rays_d, dxdu, dydv,
                    segmentation, img 
                ],-1))

                int_albedo_path = os.path.join(self.root_dir, self.albedo_dir, '{:0>3d}_0001.png'.format(img_idx))
                int_albedo = np.array(Image.open(int_albedo_path)).reshape(-1, 3)
                int_albedo = torch.from_numpy(int_albedo/255.0).float()
                self.all_intrinsic += [int_albedo]

                if self.cache_dir is not None:
                    # load diffuse and specular shadings
                    diffuse = open_exr(Path(self.cache_dir,'diffuse','{:03d}.exr'.format(idx)),img_hw).reshape(-1,3)
                    speculars0,speculars1 = [],[]
                    for r_idx in range(self.roughness_level):
                        specular0 = open_exr(Path(self.cache_dir,'specular','{:03d}_0_{}.exr'.format(idx,r_idx)),img_hw)
                        specular0 = specular0.reshape(-1,3)
                        speculars0.append(specular0)
                        specular1 = open_exr(Path(self.cache_dir,'specular','{:03d}_1_{}.exr'.format(idx,r_idx)),img_hw)
                        specular1 = specular1.reshape(-1,3)
                        speculars1.append(specular1)
                    speculars0 = torch.cat(speculars0,-1)
                    speculars1 = torch.cat(speculars1,-1)
                    self.all_cache += [torch.cat([diffuse, speculars0, speculars1,],1)]

            self.all_rays = torch.cat(self.all_rays,0)
            self.all_intrinsic = torch.cat(self.all_intrinsic, 0)
            if self.cache_dir is not None:
                self.all_cache = torch.cat(self.all_cache, 0)

            # number of pixel batches
            self.batch_num = math.ceil(len(self.all_rays)*1.0/self.batch_size)
            self.idxs = torch.randperm(len(self.all_rays))
            if self.multi_exposure:
                pixel_num = self.img_hw[0] * self.img_hw[1]
                self.exposures = torch.tensor(self.exposures).view(-1, 1, 1).repeat(1, pixel_num, 1).view(-1, 1)
    
    def resample(self,):
        """ resample pixel batch """
        self.idxs = torch.randperm(len(self.all_rays))
    
    def __len__(self,):
        if self.pixel == True:
            return self.batch_num
        # if self.split == 'val':
        #     return 8
        return len(self.C2Ws)
    
    def __getitem__(self,idx):
        exposure = None
        if self.pixel:
            # find pixel indices for current batch
            b0 = idx*self.batch_size
            b1 = min(b0+self.batch_size,len(self.all_rays))
            
            idx = self.idxs[b0:b1]
            tmp = self.all_rays[idx]
            if self.multi_exposure:
                exposure = self.exposures[idx]
            
            diffuse, specular0, specular1 = None, None, None
            if self.cache_dir is not None:
                cache = self.all_cache[idx]
                diffuse = cache[..., :3]
                specular0 = cache[..., 3:21].reshape(b1-b0,-1,3)
                specular1 = cache[..., 21:39].reshape(b1-b0,-1,3)
            
            return {
                'rays': tmp[...,:12],
                'segmentation': tmp[...,12],
                'rgbs': tmp[...,13:16],
                'int_albedo': self.all_intrinsic[idx],
                'exposure': exposure, 
                'diffuse': diffuse,
                'specular0': specular0,
                'specular1': specular1,
            }
        
        k = self.Ks[idx]
        c2w = self.C2Ws[idx]
        img_hw = self.img_hw
        img_idx = self.split_ids[idx]
        img = open_png(Path(self.root_dir,     
            self.img_dir,'{:03d}_0001.png'.format(img_idx)),img_hw).reshape(-1,3).clamp_min(0)
        
        segmentation = torch.from_numpy(cv2.imread(os.path.join(self.root_dir,'segmentation','{:03d}.exr'.format(img_idx)),-1))[...,0].reshape(-1,1).float()

        rays_d = get_direction(k,img_hw)
        rays_x,rays_d,dxdu,dydv = to_world(rays_d,c2w,self.ray_diff,k)
        rays = torch.cat([rays_x,rays_d,dxdu,dydv],-1)

        int_albedo_path = os.path.join(self.root_dir, self.albedo_dir, '{:0>3d}_0001.png'.format(img_idx))
        int_albedo = np.array(Image.open(int_albedo_path)).reshape(-1, 3)
        int_albedo = torch.from_numpy(int_albedo/255.0).float()
        if self.multi_exposure:
            exposure = self.exposures[idx]
        
        diffuse, speculars0, speculars1 = None, None, None
        if self.cache_dir is not None:
            diffuse = open_exr(Path(self.cache_dir,'diffuse','{:03d}.exr'.format(idx)),img_hw).reshape(-1,3)
            speculars0,speculars1 = [],[]
            for r_idx in range(self.roughness_level):
                specular0 = open_exr(Path(self.cache_dir,'specular','{:03d}_0_{}.exr'.format(idx,r_idx)),img_hw)
                specular0 = specular0.reshape(-1,3)
                speculars0.append(specular0)
                specular1 = open_exr(Path(self.cache_dir,'specular','{:03d}_1_{}.exr'.format(idx,r_idx)),img_hw)
                specular1 = specular1.reshape(-1,3)
                speculars1.append(specular1)
            speculars0 = torch.stack(speculars0,-2)
            speculars1 = torch.stack(speculars1,-2)
     
        return {
            'rays': rays,
            'segmentation': segmentation,
            'rgbs': img,
            'int_albedo': int_albedo,
            'exposure': exposure,
            'c2w': c2w,
            'diffuse':diffuse,
            'specular0': speculars0,
            'specular1': speculars1,
            'img_hw': img_hw,
        }