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
from PIL import Image
from torchvision import transforms as T
import cv2
import math
from const import GAMMA, set_random_seed
set_random_seed()

def get_ray_directions(H, W, focal):
    """ get camera ray direction
    Args:
        H,W: height and width
        focal: focal length
    x: left, y: up, z: forward
    """
    x_coords = torch.linspace(0.5, W - 0.5, W)
    y_coords = torch.linspace(0.5, H - 0.5, H)
    j, i = torch.meshgrid([y_coords, x_coords])
    directions = \
        torch.stack([-(i-W/2)/focal, -(j-H/2)/focal, torch.ones_like(i)], -1) 

    return directions

def get_rays(directions, c2w, focal=None):
    """ world space camera ray
    Args:
        directions: camera ray direction (local)
        c2w: 3x4 camera to world matrix
        focal: if not None, return ray differentials as well
    """
    R = c2w[:,:3]
    rays_d = directions @ R.T
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    if focal is not None:
        dxdu = torch.tensor([1.0/focal,0,0])[None,None].expand_as(directions)@R.T
        dydv = torch.tensor([0,1.0/focal,0])[None,None].expand_as(directions)@R.T
        dxdu = dxdu.view(-1,3)
        dydv = dydv.view(-1,3)
        return rays_o, rays_d, dxdu, dydv
    else:
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        return rays_o, rays_d

def open_exr(file,img_hw):
    img = cv2.imread(file,cv2.IMREAD_UNCHANGED)[...,[2,1,0]]
    assert img.shape[0] == img_hw[0]
    assert img.shape[1] == img_hw[1]
    #img = cv2.resize(img,img_hw,cv2.INTER_LANCZOS4)
    img = torch.from_numpy(img.astype(np.float32))
    return img

def open_png(file,img_hw, gamma=None):
    img = cv2.imread(str(file))[...,[2,1,0]]
    hs, ws, _ = img.shape
    ht, wt    = img_hw
    if (ht != hs) or (wt != ws):
        img = cv2.resize(img, (wt, ht))
    #img = cv2.resize(img,img_hw,cv2.INTER_LANCZOS4)
    img = torch.from_numpy((img/255).astype(np.float32))
    # convert to linear color space, the saturated pixel intensity is incorrect
    if gamma:
        img = img.pow(gamma)
    return img

class SyntheticDatasetLDR(Dataset):
    """ synthetic dataset in structure:
    Scene/
        {SPLIT}/ train or val split
            Image/{:03d}_0001.exr HDR images
            Roughness/{:03d}_0001.exr Roughness
            DiffCol/{:03d}_0001.exr diffuse reflectance
            albedo/{:03d}.exr material reflectance a'=\int f d\omega_i
            Emit/{:03d}_0001.exr emission 
            IndexMA/{:03d}_0001.exr material part segmentation
            segmentation/{:03d}.exr semantic segmentation
            transforms.json c2w camera matrix file and fov
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
            ray_diff: whether load ray differentials
        """
        self.root_dir = os.path.join(root_dir,split) if split != 'relight'\
                      else os.path.join(root_dir,'val')
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
        self.pixel=pixel
        self.split = split

        h, w = cv2.imread(os.path.join(root_dir,'train/Image/000_0001.exr'),-1).shape[:2]
        self.img_hw = (int(h*res_scale), int(w*res_scale))

        self.ray_diff = ray_diff
        self.val_frame = val_frame
        
        with open(os.path.join(self.root_dir,
                               f"transforms.json"), 'r') as f:
            self.meta = json.load(f)

        # camera focal length and ray directions
        h,w = self.img_hw
        self.focal = (0.5*w/np.tan(0.5*self.meta['camera_angle_x'])).item()
        self.directions = \
            get_ray_directions(h, w, self.focal)
        
        # load every camera pixels
        if self.pixel:
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for cur_idx in range(len(self.meta['frames'])):
                frame = self.meta['frames'][cur_idx]
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)
                
                image_path = os.path.join(self.root_dir, self.img_dir, '{:03d}_0001.png'.format(cur_idx))
                img = open_png(image_path,self.img_hw,self.gamma).reshape(-1,3)
                
                # load ground truth BRDF
                albedo = open_exr(os.path.join(self.root_dir,'DiffCol','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
                roughness = open_exr(os.path.join(self.root_dir,'Roughness','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)[...,:1]
                emission = open_exr(os.path.join(self.root_dir,'Emit','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
  
                self.all_rgbs += [img]
                
                if self.ray_diff==False:
                    rays_o, rays_d = get_rays(self.directions, c2w) 

                    self.all_rays += [torch.cat([rays_o, rays_d,
                                                 albedo,
                                                 roughness,
                                                 emission
                                                ],1)]
                else:
                    rays_o,rays_d,dxdu,dydv = get_rays(self.directions,c2w,focal=self.focal)
                    self.all_rays += [torch.cat([rays_o, rays_d,
                                                 dxdu,dydv,
                                                 albedo,
                                                 roughness,
                                                 emission
                                                ],1)]

            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            if self.multi_exposure:
                pixel_num = self.img_hw[0] * self.img_hw[1]
                self.exposures = torch.tensor(self.exposures).view(-1, 1, 1).repeat(1, pixel_num, 1).view(-1, 1)
        
        if load_traj:
            render_traj = np.load(os.path.join(root_dir, 'render_traj.npy'))
            self.render_traj_c2w = render_traj
            self.render_traj_rays = []
            for i in range(len(render_traj)):
                c2w = torch.FloatTensor(render_traj[i])
                rays_o,rays_d,dxdu,dydv = get_rays(self.directions,c2w,focal=self.focal)
                self.render_traj_rays += [torch.cat([rays_o, rays_d, dxdu, dydv],1)]


    def __len__(self):
        if self.pixel==True:
            return len(self.all_rays)
        # if self.split == 'val':
        #     # only show 8 images for reconstruction validation
        #     return 8
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        exposure = None
        if self.pixel: 
            if self.multi_exposure:
                exposure = self.exposures[idx]
            tmp = self.all_rays[idx]
            if self.ray_diff == False:
                sample = {'rays': tmp[:8],
                          'rgbs': self.all_rgbs[idx],
                          'albedo': tmp[8:11],
                          'roughness': tmp[11],
                          'emission': tmp[12:15],
                          'exposure': exposure
                         }
            else:
                sample = {'rays': tmp[:12],
                      'rgbs': self.all_rgbs[idx],
                      'albedo': tmp[12:15],
                      'roughness': tmp[15],
                      'emission': tmp[16:19],
                      'exposure': exposure
                     }

        else:
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            cur_idx = idx
            
            image_path = os.path.join(self.root_dir, self.img_dir,'{:03d}_0001.png'.format(cur_idx))
            img = open_png(image_path,self.img_hw,self.gamma).reshape(-1,3)
            
            albedo = open_exr(os.path.join(self.root_dir,'DiffCol','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
            roughness = open_exr(os.path.join(self.root_dir,'Roughness','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)[...,0]
            emission = open_exr(os.path.join(self.root_dir,'Emit','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
            
            if self.ray_diff == False:
                rays_o, rays_d = get_rays(self.directions, c2w)

                rays = torch.cat([rays_o, rays_d],1)
            else:
                rays_o, rays_d,dxdu,dydv = get_rays(self.directions, c2w,focal=self.focal)
                rays = torch.cat([rays_o, rays_d,dxdu,dydv],1)
            if self.multi_exposure:
                exposure = self.exposures[idx]
            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'albedo':albedo,
                      'roughness': roughness,
                      'emission': emission,
                      'exposure': exposure
                     }

        return sample

class InvSyntheticDatasetLDR(Dataset):
    """ Synthetic dataset with diffuse and specular shadings
    Shading folder in structure:
    Scene/
        diffuse/{:03d}.exr diffuse shadings (L_d)
        specular/{:03d}_i_j.exr specular shadings (L_s^i(\sigma_j))
    """
    def __init__(self, root_dir, img_dir=None, batch_size=None,split='train', 
                 pixel=True,has_part=False, load_metallic=False, val_frame=0, cache_dir=None):
        """
        Args:
            root_dir: dataset root folder
            cache_dir: shadings folder
            split: train or val
            pixel: whether load every camera pixel
            batch_size: size of each ray batch if pixel==True
            has_part: whether use ground truth part segmentation or not (semantic segmentation)
        """
        self.root_dir = os.path.join(root_dir,split)
        self.cache_dir = cache_dir
        if img_dir == None:
            self.img_dir = 'Image'
            self.albedo_dir = 'irisformer/albedo'
            self.exposures = None 
            self.crfs      = None
            self.multi_exposure = False
            self.gamma = GAMMA
        else:
            self.img_dir = img_dir 
            self.albedo_dir = os.path.join(img_dir, 'albedo')
            self.exposures = np.load(os.path.join(self.root_dir, self.img_dir, 'cam', 'exposure.npy'))
            self.crfs      = np.load(os.path.join(self.root_dir, self.img_dir, 'cam', 'crf.npy')) #(3, 1024)
            self.multi_exposure = True
            self.gamma = None       
        self.pixel=pixel
        self.split = split
        self.batch_size = batch_size
        self.has_part = has_part
        self.load_metallic = load_metallic
        self.val_frame = val_frame
        # approximate roughness channel by interpolating 6 samples
        self.roughness_level = 6
        
        self.img_hw = cv2.imread(os.path.join(root_dir,'train/Image/000_0001.exr'),-1).shape[:2]
        

        with open(os.path.join(self.root_dir,
                               f"transforms.json"), 'r') as f:
            self.meta = json.load(f)

        h,w = self.img_hw
        self.focal = (0.5*w/np.tan(0.5*self.meta['camera_angle_x'])).item()
        self.directions = get_ray_directions(h, w, self.focal)
            
        if self.pixel: 
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_intrinsic = []
            self.all_cache = []
            for cur_idx in range(len(self.meta['frames'])):
                frame = self.meta['frames'][cur_idx]
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)
                
                image_path = os.path.join(self.root_dir, self.img_dir, '{:03d}_0001.png'.format(cur_idx))
                img = open_png(image_path,self.img_hw,self.gamma).reshape(-1,3)

                # ground truth brdf-emission
                albedo = open_exr(os.path.join(self.root_dir,'DiffCol','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
                roughness = open_exr(os.path.join(self.root_dir,'Roughness','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)[:,:1]
                emission = open_exr(os.path.join(self.root_dir,'Emit','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)

                # load part or semantic segmentation
                if self.has_part:
                    segmentation = torch.from_numpy(cv2.imread(os.path.join(self.root_dir,'IndexMA','{:03d}_0001.exr'.format(cur_idx)),-1))[...,0].reshape(-1,1).float()
                else:
                    segmentation = torch.from_numpy(cv2.imread(os.path.join(self.root_dir,'segmentation','{:03d}.exr'.format(cur_idx)),-1))[...,0].reshape(-1,1).float()

                self.all_rgbs += [img]
                rays_o, rays_d,dxdu,dydv = get_rays(self.directions, c2w, focal=self.focal) # both (h*w, 3)

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             dxdu,
                                             dydv,
                                             albedo,
                                             roughness,
                                             emission,
                                             segmentation
                                            ],1)] 
                
                int_albedo_path = os.path.join(self.root_dir, self.albedo_dir, '{:0>3d}_0001.png'.format(cur_idx))
                int_albedo = np.array(Image.open(int_albedo_path)).reshape(-1, 3)
                int_albedo = torch.from_numpy(int_albedo/255.0).float()
                self.all_intrinsic += [int_albedo]

                if self.cache_dir is not None:
                    # load shadings
                    diffuse = open_exr(os.path.join(self.cache_dir,'diffuse','{:03d}.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
                    speculars0,speculars1 = [],[]
                    for r_idx in range(self.roughness_level):
                        specular0 = open_exr(os.path.join(self.cache_dir,'specular','{:03d}_0_{}.exr'.format(cur_idx,r_idx)),self.img_hw).float()
                        specular0 = specular0.reshape(-1,3)
                        speculars0.append(specular0)
                        specular1 = open_exr(os.path.join(self.cache_dir,'specular','{:03d}_1_{}.exr'.format(cur_idx,r_idx)),self.img_hw).float()
                        specular1 = specular1.reshape(-1,3)
                        speculars1.append(specular1)
                    speculars0 = torch.cat(speculars0,-1)
                    speculars1 = torch.cat(speculars1,-1)
                    self.all_cache += [torch.cat([diffuse, speculars0, speculars1,],1)]

            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            self.all_intrinsic = torch.cat(self.all_intrinsic, 0)
            if self.cache_dir is not None:
                self.all_cache = torch.cat(self.all_cache, 0)

            # number of camera ray batches
            self.batch_num = math.ceil(len(self.all_rays)*1.0/self.batch_size)
            self.idxs = torch.randperm(len(self.all_rays))
            if self.multi_exposure:
                pixel_num = self.img_hw[0] * self.img_hw[1]
                self.exposures = torch.tensor(self.exposures).view(-1, 1, 1).repeat(1, pixel_num, 1).view(-1, 1)
            if load_metallic:
                metallic_all = np.load(os.path.join(self.root_dir, 'metallic.npy'))
                self.metallic_all = metallic_all.reshape(-1)
    
    def resample(self,):
        # resample camera ray batches
        self.idxs = torch.randperm(len(self.all_rays))

    def __len__(self):
        if self.pixel==True:
            return self.batch_num
        # if self.split == 'val':
        #     return 8
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        exposure = None
        if self.pixel:
            b0 = idx*self.batch_size
            b1 = min(b0+self.batch_size,len(self.all_rays))
            
            # find camera ray indices in the batch
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
            
            sample = {'rays': tmp[...,:12],
                      'albedo': tmp[...,12:15],
                      'roughness': tmp[...,15],
                      'emission': tmp[...,16:19],
                      'segmentation': tmp[...,19],
                      'rgbs': self.all_rgbs[idx],
                      'int_albedo': self.all_intrinsic[idx],
                      'exposure': exposure,
                      'diffuse': diffuse,
                      'specular0': specular0,
                      'specular1': specular1
                    }
            if self.load_metallic:
                sample['metallic'] = self.metallic_all[idx]
            

        else:
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]
            cur_idx = idx
            
            image_path = os.path.join(self.root_dir, self.img_dir, '{:03d}_0001.png'.format(cur_idx))
            img = open_png(image_path,self.img_hw,self.gamma).reshape(-1,3)
            
            albedo = open_exr(os.path.join(self.root_dir,'DiffCol','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
            roughness = open_exr(os.path.join(self.root_dir,'Roughness','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)[:,0]
            emission = open_exr(os.path.join(self.root_dir,'Emit','{:03d}_0001.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)

            if self.has_part:
                segmentation = torch.from_numpy(cv2.imread(os.path.join(self.root_dir,'IndexMA','{:03d}_0001.exr'.format(cur_idx)),-1))[...,0].reshape(-1).float()
            else:
                segmentation = torch.from_numpy(cv2.imread(os.path.join(self.root_dir,'segmentation','{:03d}.exr'.format(cur_idx)),-1))[...,0].reshape(-1).float()
            
            rays_o,rays_d,dxdu,dydv = get_rays(self.directions, c2w, focal=self.focal)

            rays = torch.cat([rays_o, rays_d,
                              dxdu,
                              dydv],-1)
            
            int_albedo_path = os.path.join(self.root_dir, self.albedo_dir, '{:0>3d}_0001.png'.format(cur_idx))
            int_albedo = np.array(Image.open(int_albedo_path)).reshape(-1, 3)
            int_albedo = torch.from_numpy(int_albedo/255.0).float()
            if self.multi_exposure:
                exposure = self.exposures[idx]

            diffuse, speculars0, speculars1 = None, None, None
            if self.cache_dir is not None:
                diffuse = open_exr(os.path.join(self.cache_dir,'diffuse','{:03d}.exr'.format(cur_idx)),self.img_hw).reshape(-1,3)
                speculars0,speculars1 = [],[]
                for r_idx in range(self.roughness_level):
                    specular0 = open_exr(os.path.join(self.cache_dir,'specular','{:03d}_0_{}.exr'.format(cur_idx,r_idx)),self.img_hw).float()
                    specular0 = specular0.reshape(-1,3)
                    speculars0.append(specular0)
                    specular1 = open_exr(os.path.join(self.cache_dir,'specular','{:03d}_1_{}.exr'.format(cur_idx,r_idx)),self.img_hw).float()
                    specular1 = specular1.reshape(-1,3)
                    speculars1.append(specular1)
                speculars0 = torch.stack(speculars0,-2)
                speculars1 = torch.stack(speculars1,-2)
            
            sample = {
                'rays': rays,
                'rgbs': img,
                'c2w': c2w,
                'albedo':albedo,
                'roughness': roughness,
                'emission': emission,
                'segmentation': segmentation,
                'int_albedo': int_albedo,
                'exposure': exposure,
                'diffuse':diffuse,
                'specular0': speculars0,
                'specular1': speculars1,
            }
            if self.load_metallic:
                metallic_all = np.load(os.path.join(self.root_dir, 'metallic.npy'))
                metallic = metallic_all[cur_idx].reshape(-1)
                sample['metallic'] = metallic
        return sample