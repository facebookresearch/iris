# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as NF

import math
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm

METHOD = 'fipt_syn_kitchen'
GT_PATH = '/hdd/datasets/fipt/indoor_synthetic/kitchen/train'
METHOD_PATH = os.path.join('outputs', METHOD, 'output', 'train')

image_num = len([f for f in os.listdir(os.path.join(GT_PATH,'Image')) if f[0] != '.' and f[-4:]=='.exr'])
mse_roughness = []
mse_albedo = []
mse_diff = []
iou_emission = []
mse_emission = []
for i in tqdm(range(image_num)):
    emission_gt = cv2.imread(os.path.join(GT_PATH,'Emit','{:03d}_0001.exr'.format(i)),-1)[...,[2,1,0]]
    emission_gt = torch.from_numpy(emission_gt).float()
    emission_mask = emission_gt.sum(-1) > 0
    albedo_gt = cv2.imread(os.path.join(GT_PATH,'albedo','{:03d}.exr'.format(i)),-1)[...,[2,1,0]]
    albedo_gt = torch.from_numpy(albedo_gt).float().clamp(0,1).mul(255).long().float()/255
    albedo_gt[emission_mask] = 0
    
    kd_gt = cv2.imread(os.path.join(GT_PATH,'DiffCol','{:03d}_0001.exr'.format(i)),-1)[...,[2,1,0]]
    kd_gt = torch.from_numpy(kd_gt).float().clamp(0,1).mul(255).long().float()/255
    kd_gt[emission_mask] = 0
    
    roughness_gt = cv2.imread(os.path.join(GT_PATH,'Roughness','{:03d}_0001.exr'.format(i)),-1)[...,0]
    roughness_gt = (torch.from_numpy(roughness_gt).float().mul(255).long().float()/255).clamp(0.2,1)
    roughness_gt[emission_mask] = 0
    
    diff_mask =roughness_gt==1
    kd_gt[~diff_mask] = 0
    
    
    emission = cv2.imread(os.path.join(METHOD_PATH, 'emission', '{:05d}_emission.exr'.format(i)),-1)[...,[2,1,0]]
    emission = torch.from_numpy(emission).float()
    
    albedo = cv2.imread(os.path.join(METHOD_PATH, 'a_prime','{:05d}_a_prime.png'.format(i)),-1)[...,[2,1,0]]
    albedo = torch.from_numpy(albedo).float()/255
    albedo[emission_mask] = 0
    
    kd = cv2.imread(os.path.join(METHOD_PATH, 'diffuse', '{:05d}_kd.png'.format(i)),-1)[...,[2,1,0]]
    kd = torch.from_numpy(kd).float()/255
    kd[emission_mask] = 0
    kd[~diff_mask] = 0
    
    roughness = cv2.imread(os.path.join(METHOD_PATH, 'roughness','{:05d}_roughness.png'.format(i)),-1)
    roughness = roughness[:, :, 0]
    roughness = (torch.from_numpy(roughness).float()/255).clamp(0.2,1)
    
    roughness[emission_mask] = 0

    
    emission_mask_est = emission.sum(-1)>0
    if emission_mask.any():
        iou = (emission_mask&emission_mask_est).sum()*1.0/(emission_mask|emission_mask_est).sum()
        iou_emission.append(iou)
        intersect = emission_mask&emission_mask_est
        mse_emission.append(NF.mse_loss(torch.log(emission+1),torch.log(emission_gt+1)))
        #mse_emission.append(NF.mse_loss(emission.pow(1/2.2),emission_gt.pow(1/2.2)))
    
    mse_roughness.append(NF.mse_loss(roughness,roughness_gt))
    mse_albedo.append(NF.mse_loss(albedo,albedo_gt))
    mse_diff.append(NF.mse_loss(kd,kd_gt))
mse_roughness = torch.tensor(mse_roughness)
mse_diff = torch.tensor(mse_diff)
mse_albedo = torch.tensor(mse_albedo)
mse_emission = torch.tensor(mse_emission)
iou_emission = torch.tensor(iou_emission)

print(METHOD)
print('kd:          ',-10*torch.log10(mse_diff.clamp_min(1e-10)).mean().item())
print('albedo:      ',-10*torch.log10(mse_albedo.clamp_min(1e-10)).mean().item())
print('roughness:   ',(-10*torch.log10(mse_roughness.clamp_min(1e-10))).mean().item())
print('emit_iou:    ',iou_emission.mean().item())
print('emit_log_mse:',mse_emission.mean().item())