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
from utils.dataset.real_ldr import normalize_v, read_cam_params, open_exr, get_direction, to_world
from utils.dataset.synthetic_ldr import open_png
from crf.emor import parse_emor_file
from .colmap_utils import read_images_text
from PIL import Image
from const import GAMMA, set_random_seed
set_random_seed()

def get_split_ids(n_total, split='train'):
    val_ids = [i*10 for i in range(16)]
    train_ids = [i for i in range(n_total) if i not in val_ids]
    if split == 'train':
        return train_ids
    else:
        return val_ids 

def read_json(path):
    with open(path) as f:
        content = json.load(f)
        return content
    
def sample_nearest(array, hw):
    img = Image.fromarray(array)
    h, w = hw
    img = img.resize((w, h), Image.NEAREST)
    array = np.array(img)
    return array
class Scannetpp(Dataset):
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
                 scene_id, 
                 split='all', 
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
        self.pixel = pixel
        self.split = split
        self.ray_diff = ray_diff
        self.val_frame = val_frame
        self.multi_exposure = True
        self.gamma = None

        self.dir_scene  = os.path.join(root_dir, 'data', scene_id, 'psdf')
        self.dir_rgb    = os.path.join(self.dir_scene, 'images')
        # self.dir_mask   = os.path.join(self.dir_scene, 'undistorted_anon_masks')
        # self.dir_albedo = os.path.join(self.dir_scene, 'albedo')

        train_test_list = read_json(os.path.join(self.dir_scene, 'train_test_lists.json'))
        if split == 'train':
            names = train_test_list['train']
        elif split == 'test':
            names = train_test_list['test']
        else:
            names = train_test_list['train'] + train_test_list['test']
        self.img_name_list = names

        self.exposures = np.ones(len(names)).astype(np.float32)
        _, vectors = parse_emor_file(inv=False)
        crf_mean = vectors[1]
        self.crfs = np.stack([crf_mean, crf_mean, crf_mean]).astype(np.float32)

        # camera intrinsics
        transform = read_json(os.path.join(self.dir_scene, 'transforms_all.json'))
        fx, fy = transform['fl_x'], transform['fl_y']
        cx, cy = transform['cx'], transform['cy']
        h, w = transform['h'], transform['w']
        self.img_hw = (int(h*res_scale), int(w*res_scale))
        K = np.array([
            [fx, 0, cx], 
            [0, fy, cy],
            [0, 0 ,1]
        ])
        K[:2] *= res_scale
        Ks = [K for i in range(len(names))]
        self.Ks = torch.tensor(Ks).float()

        # camera extrinsics, load from colmap files which is aligned with scans
        # images_path = os.path.join(self.dir_scene, 'colmap', 'images.txt')
        # imdata = read_images_text(images_path)
        # img_names = [imdata[k].name for k in imdata]
        # w2c_mats = []
        # bottom = np.array([[0, 0, 0, 1.]])
        # for k in imdata:
        #     im = imdata[k]
        #     R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
        #     w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        # w2c_mats = np.stack(w2c_mats, 0)
        # c2w = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4) cam2world matrices
        # c2w_dict = {img_names[i]: c2w[i] for i in range(len(img_names))}
        # c2ws = np.array([c2w_dict[name] for name in names])

        frames = transform['frames']
        ids = []
        c2ws = []
        for frame in frames:
            name = frame['file_path'].split('/')[-1]
            if name not in self.img_name_list:
                continue
            ids.append(self.img_name_list.index(name))
            c2w = np.array(frame['transform_matrix'])
            c2w[:3, 1:3] *= -1 # to OpenCV
            c2ws.append(c2w[:3])
        ids = np.array(ids)
        c2ws = np.array(c2ws)
        argsort = np.argsort(ids)
        c2ws = c2ws[argsort]

        self.C2Ws = torch.from_numpy(c2ws).float()
        # print(img.qvec2rotmat())

        if self.pixel:
            # load all camera pixels
            self.all_rays = []
            for idx in range(len(self.img_name_list)):
                k = self.Ks[idx]
                c2w = self.C2Ws[idx]
                img_hw = self.img_hw
                img_name = self.img_name_list[idx]
                img = open_png(Path(self.dir_rgb, img_name),img_hw,self.gamma).reshape(-1,3).clamp_min(0)
                
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
            render_traj = np.load(os.path.join(root_dir, 'data', scene_id, 'psdf', 'render_traj.npy'))
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

        img_name = self.img_name_list[idx]
        img = open_png(Path(self.dir_rgb, img_name),img_hw,self.gamma).reshape(-1,3).clamp_min(0)
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
    
    
    
class InvScannetpp(Dataset):
    """ Real world capture dataset with diffuse and specular shadings
    Shading folder in structure:
    Scene/
        diffuse/{:03d}.exr diffuse shadings (L_d)
        specular/{:03d}_i_j.exr specular shadings (L_s^i(\sigma_j))
    """
    def __init__(self, 
                 root_dir, 
                 scene_id, 
                 split='all',
                 batch_size=1000,
                 pixel=True, 
                 val_frame=0, 
                 cache_dir=None,
                 res_scale=1.0):
        """
        Args:
            root_dir: dataset root folder
            cache_dir: shadings folder
            split: train or val
            pixel: whether load every camera pixel
            batch_size: size of each ray batch if pixel==True
        """
        self.root_dir = root_dir
        self.pixel = pixel
        self.split = split
        self.batch_size = batch_size
        self.val_frame = val_frame
        self.cache_dir = cache_dir
        self.multi_exposure = True
        self.gamma = None 
        self.ray_diff = True
        self.roughness_level = 6

        self.dir_scene  = os.path.join(root_dir, 'data', scene_id, 'psdf')
        self.dir_rgb    = os.path.join(self.dir_scene, 'images')
        # self.dir_mask   = os.path.join(self.dir_scene, 'undistorted_anon_masks')
        self.dir_albedo = os.path.join(self.dir_scene, 'albedo')
        self.dir_seg    = os.path.join(self.dir_scene, 'seg')

        train_test_list = read_json(os.path.join(self.dir_scene, 'train_test_lists.json'))
        if split == 'train':
            names = train_test_list['train']
        elif split == 'test':
            names = train_test_list['test']
        else:
            names = train_test_list['train'] + train_test_list['test']
        self.img_name_list = names

        self.exposures = np.ones(len(names)).astype(np.float32)
        _, vectors = parse_emor_file(inv=False)
        crf_mean = vectors[1]
        self.crfs = np.stack([crf_mean, crf_mean, crf_mean]).astype(np.float32)

        # camera intrinsics
        transform = read_json(os.path.join(self.dir_scene, 'transforms_all.json'))
        fx, fy = transform['fl_x'], transform['fl_y']
        cx, cy = transform['cx'], transform['cy']
        h, w = transform['h'], transform['w']
        self.img_hw = (int(h*res_scale), int(w*res_scale))
        K = np.array([
            [fx, 0, cx], 
            [0, fy, cy],
            [0, 0 ,1]
        ])
        K[:2] *= res_scale
        Ks = [K for i in range(len(names))]
        self.Ks = torch.tensor(Ks).float()

        # camera extrinsics, load from colmap files which is aligned with scans
        # images_path = os.path.join(self.dir_scene, 'colmap', 'images.txt')
        # imdata = read_images_text(images_path)
        # img_names = [imdata[k].name for k in imdata]
        # w2c_mats = []
        # bottom = np.array([[0, 0, 0, 1.]])
        # for k in imdata:
        #     im = imdata[k]
        #     R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
        #     w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        # w2c_mats = np.stack(w2c_mats, 0)
        # c2w = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4) cam2world matrices
        # c2w_dict = {img_names[i]: c2w[i] for i in range(len(img_names))}
        # c2ws = np.array([c2w_dict[name] for name in names])

        frames = transform['frames']
        ids = []
        c2ws = []
        for frame in frames:
            name = frame['file_path'].split('/')[-1]
            if name not in self.img_name_list:
                continue
            ids.append(self.img_name_list.index(name))
            c2w = np.array(frame['transform_matrix'])
            c2w[:3, 1:3] *= -1 # to OpenCV
            c2ws.append(c2w[:3])
        ids = np.array(ids)
        c2ws = np.array(c2ws)
        argsort = np.argsort(ids)
        c2ws = c2ws[argsort]

        self.C2Ws = torch.from_numpy(c2ws).float()
        
        if self.pixel:
            self.all_rays = []
            self.all_intrinsic = []
            self.all_cache = []
            for idx in range(len(self.Ks)):
                k = self.Ks[idx]
                c2w = self.C2Ws[idx]
                img_hw = self.img_hw
                img_name = self.img_name_list[idx]
                img = open_png(Path(self.dir_rgb, img_name),img_hw,self.gamma).reshape(-1,3).clamp_min(0)

                # segmentation mask
                seg = cv2.imread(os.path.join(self.dir_seg, img_name))[:, :, 0]
                if seg.shape[0] != img_hw[0]:
                    seg = sample_nearest(seg, img_hw)
                segmentation = torch.from_numpy(seg).reshape(-1,1).float()
                
                rays_d = get_direction(k,img_hw)
                rays_x,rays_d,dxdu,dydv = to_world(rays_d,c2w,self.ray_diff,k)
                self.all_rays.append(torch.cat([
                    rays_x, rays_d, dxdu, dydv,
                    segmentation, img 
                ],-1))

                int_albedo_path = os.path.join(self.dir_albedo, img_name)
                int_albedo = open_png(int_albedo_path, img_hw).reshape(-1, 3)
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
        img_name = self.img_name_list[idx]
        img = open_png(Path(self.dir_rgb, img_name),img_hw,self.gamma).reshape(-1,3).clamp_min(0)
        
        seg = cv2.imread(os.path.join(self.dir_seg, img_name))[:, :, 0]
        if seg.shape[0] != img_hw[0]:
            seg = sample_nearest(seg, img_hw)
        segmentation = torch.from_numpy(seg).reshape(-1,1).float()

        rays_d = get_direction(k,img_hw)
        rays_x,rays_d,dxdu,dydv = to_world(rays_d,c2w,self.ray_diff,k)
        rays = torch.cat([rays_x,rays_d,dxdu,dydv],-1)

        int_albedo_path = os.path.join(self.dir_albedo, img_name)
        int_albedo = open_png(int_albedo_path, img_hw).reshape(-1, 3)
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
    
def test_scannetpp():
    root_dir = '/hdd/datasets/scannetpp/'
    scene_id = '45b0dac5e3'
    split = 'train'
    pixel = False
    dataset = Scannetpp(root_dir, scene_id, split, pixel)
    
    print('len:', len(dataset))
    print('c2ws:', dataset.C2Ws.shape)
    print('Ks:', dataset.Ks.shape)
    print('exposure:', dataset.exposures.shape)

    idx = 0
    print(dataset.img_name_list[idx])
    c2w = dataset.C2Ws[idx]
    c2w[:, 1:3] *= -1
    m = np.eye(4)
    m[:3] = c2w
    print('matrix:')
    ml = []
    for i in range(len(m)):
        ml.append(list(m[i]))
    print(ml)
    # idx = 0
    # batch = dataset[idx]
    # for k in ['rays', 'rgbs', 'exposure']:
    #     print('{}: {}'.format(k, batch[k].shape))
    # print('exposure:', batch['exposure'])
    # print('img_hw:', batch['img_hw'])
        
def test_invscannetpp():
    dataset = InvScannetpp(
        root_dir='/hdd/datasets/scannetpp/', 
        scene_id='45b0dac5e3', 
        split='train',
        batch_size=1024,
        pixel=False, 
        val_frame=0, 
        cache_dir=None,
        res_scale=0.5
    )
    
    print('len:', len(dataset))
    print('c2ws:', dataset.C2Ws.shape)
    print('Ks:', dataset.Ks.shape)
    print('exposure:', dataset.exposures.shape)

    idx = 0
    batch = dataset[idx]
    for k in batch.keys():
        print('{}: {}'.format(k, batch[k].shape))

if __name__ == '__main__':
    test_scannetpp()