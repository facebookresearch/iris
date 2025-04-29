# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import skimage
import imageio
from pathlib import Path
from configs.config import default_options
from utils.dataset import RealDatasetLDR,SyntheticDatasetLDR
from utils.dataset.scannetpp.dataset import Scannetpp
from utils.ops import *
from utils.path_tracing import ray_intersect,path_tracing,path_tracing_single

from model.brdf import NGPBRDF
from model.emitter import SLFEmitter, AreaEmitter
from crf.model_crf import EmorCRF
from crf.plot import plot_crfs

from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from argparse import Namespace, ArgumentParser
from const import GAMMA, set_random_seed
set_random_seed()

def save_image(image, path, colormap=False):
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    image = np.clip(image, 0.0, 1.0)
    image = (image*255).astype(np.uint8)
    if colormap:
        image = cv2.applyColorMap(image, cv2.COLORMAP_MAGMA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.save(path)

def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            if(args['type'] == bool):
                parser.add_argument('--{}'.format(name), type=eval, choices=[True, False], default=str(args.get('default')))
            else:
                parser.add_argument('--{}'.format(name), **args)
        return parser

def main():
    parser = ArgumentParser()
    parser = add_model_specific_args(parser)

    # add PROGRAM level args
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--output_path', type=str, default='outputs/kitchen_output')
    parser.add_argument('--device', type=int, required=False,default=0)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--ckpt', type=str, default='last.ckpt')
    parser.add_argument('--light_type', type=str, default='slf', choices=['slf', 'area'])
    parser.set_defaults(resume=False)
    args = parser.parse_args()
    args.gpus = [args.device]
    experiment_name = args.experiment_name
    device = torch.device(args.device)
    print('==========================')
    print('Exp:', args.experiment_name)
    print('Output:', args.output_path)
    print('Split:', args.split)
    print('==========================')

    dataset_name, dataset_path = args.dataset
    if dataset_name == 'synthetic':
        dataset = SyntheticDatasetLDR(dataset_path,img_dir=args.ldr_img_dir,split=args.split,pixel=False,ray_diff=True)
    elif dataset_name == 'real':
        dataset = RealDatasetLDR(dataset_path,img_dir=args.ldr_img_dir,split=args.split,pixel=False,ray_diff=True)
    elif dataset_name == 'scannetpp':
        dataset = Scannetpp(dataset_path, args.scene, pixel=False, split=args.split, ray_diff=True, res_scale=args.res_scale)
    img_hw = dataset.img_hw

    # load geometry
    if dataset_name in ['synthetic', 'real']:
        mesh_path = os.path.join(dataset_path,'scene.obj')
        mesh_type = 'obj'
    elif dataset_name == 'scannetpp':
        mesh_path = os.path.join(dataset_path, 'data', args.scene, 'scans', 'scene.ply')
        mesh_type = 'ply'
    assert Path(mesh_path).exists(), 'mesh not found: '+mesh_path
    
    scene = mitsuba.load_dict({
        'type': 'scene',
        'shape_id':{
            'type': mesh_type,
            'filename': mesh_path, 
        }
    })

    model_list = []
    # load BRDF and emitters
    emitter_path = args.emitter_path
    mask = torch.load(os.path.join(emitter_path,'vslf.npz'),map_location='cpu')
    last_ckpt = Path(args.checkpoint_path) / experiment_name / args.ckpt
    state_dict = torch.load(last_ckpt, map_location='cpu')['state_dict']
    weight = {}
    for k,v in state_dict.items():
        if 'material.' in k:
            weight[k.replace('material.','')]=v
    material_net = NGPBRDF(mask['voxel_min'],mask['voxel_max'])
    material_net.load_state_dict(weight)
    material_net.to(device)
    model_list.append(material_net)

    if args.light_type == 'slf':
        emitter_net = SLFEmitter(os.path.join(emitter_path,'emitter.pth'),
                                os.path.join(emitter_path,'vslf_0.npz'))
    else:
        emitter_net = AreaEmitter(os.path.join(emitter_path,'emitter_relight.pth'))
    emitter_net.to(device)
    model_list.append(emitter_net)

    model_crf = EmorCRF(args.crf_basis)
    weight = {}
    for k,v in state_dict.items():
        if 'model_crf.' in k:
            weight[k.replace('model_crf.','')]=v
    model_crf.load_state_dict(weight)
    model_crf.to(device)
    model_list.append(model_crf)
    
    for model in model_list:
        for p in model.parameters():
            p.requires_grad = False
    
    # create folders
    dir_out = {}
    for name in ['rgb', 'diffuse', 'a_prime', 'roughness', 'metallic', 'emission', 'slf', 'merge']:
        d = Path(args.output_path) / args.split / name
        d.mkdir(exist_ok=True, parents=True)
        dir_out[name] = d

    # set up denoiser
    denoiser = mitsuba.OptixDenoiser(img_hw[::-1])

    psnr_list = []
    ssim_list = []
    SPP = args.SPP
    spp = args.spp
    for i in tqdm(range(len(dataset))):
        batch = dataset[i]
        rays = batch['rays'].to(device)
        rays_x = rays[..., :3]
        rays_d = rays[..., 3:6]
        dxdu,dydv = rays[...,6:9],rays[...,9:12]

        L_full = torch.zeros_like(rays_x)
        kd = torch.zeros_like(rays_x)
        a_prime = torch.zeros_like(rays_x)
        roughness = torch.zeros_like(rays_x[..., :1])
        metallic = torch.zeros_like(rays_x[..., :1])
        emission = torch.zeros_like(rays_x)
        slf = torch.zeros_like(rays_x)
        for _ in range(SPP//spp):
            # render color with path tracing
            L_full += path_tracing(
                scene,emitter_net,material_net,
                rays_x,rays_d,dxdu,dydv,spp,
                indir_depth=5)
            
            # sample pixels
            du,dv = torch.rand(2,len(rays_x),spp,1,device=device)
            ds = rays_d[:,None]+ dxdu[:,None]*du + dydv[:,None]*dv
            ds = F.normalize(ds,dim=-1).reshape(-1,3)
            xs = rays_x.repeat_interleave(spp,dim=0)

            positions,normals,_,triagnle_idxs,valid = ray_intersect(scene,xs,ds)

            mat = material_net(positions)

            # get brdf parameters
            albedo_ = mat['albedo']
            metallic_ = mat['metallic']
            roughness_ = mat['roughness']
            kd_ = albedo_*(1-metallic_)
            ks_ = 0.04*(1-metallic_) + albedo_*metallic_

            # calculate material reflectance
            _,_,g0,g1 = material_net.sample_specular(
                torch.rand(len(metallic_),2,device=device),-ds,normals,roughness_)
            a_prime_ = g0*ks_+g1+kd_

            # get emission
            emission_ = emitter_net.eval_emitter(positions,ds,triagnle_idxs)[0]
            non_emit_mask = emission_.sum(-1)==0

            # get SLF
            slf_ = emitter_net(positions)

            # Set default values for emitter region
            valid = torch.logical_and(valid, non_emit_mask)
            kd_[~valid] = 1.0
            a_prime_[~valid] = 1.0
            roughness_[~valid] = 1.0
            metallic_[~valid] = 0.0

            # scene intrinsics
            kd += kd_.reshape(-1,spp,3).mean(1)
            a_prime += a_prime_.reshape(-1,spp,3).mean(1)
            roughness += roughness_.reshape(-1,spp,1).mean(1)
            metallic += metallic_.reshape(-1,spp,1).mean(1)
            emission += emission_.reshape(-1,spp,3).mean(1)
            slf += slf_.reshape(-1, spp, 3).mean(1)

        L_full = L_full.reshape(*img_hw,-1).cpu()/(SPP//spp)
        L_full = denoiser(L_full.numpy()).numpy()
        path = dir_out['rgb'] / '{:0>5d}_rgb_full.exr'.format(i)
        imageio.imwrite(path, L_full)

        exposure = batch['exposure']
        L_ldr = torch.tensor(L_full).reshape(-1, 3).to(device)
        L_ldr = model_crf(L_ldr, exposure)
        L_ldr = L_ldr.detach().reshape(*img_hw, -1).cpu().numpy()

        path = dir_out['rgb'] / '{:0>5d}_rgb_full.png'.format(i)
        save_image(L_ldr, path)
        
        L_gt = batch['rgbs'].reshape(*img_hw, -1).numpy()
        psnr = skimage.metrics.peak_signal_noise_ratio(L_gt, L_ldr, data_range=1)
        ssim = skimage.metrics.structural_similarity(L_gt, L_ldr, data_range=1, channel_axis=-1)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        kd = kd.reshape(*img_hw,-1).cpu().numpy()/(SPP//spp)
        path = dir_out['diffuse'] / '{:0>5d}_kd.exr'.format(i)
        imageio.imwrite(path, kd)
        path = dir_out['diffuse'] / '{:0>5d}_kd.png'.format(i)
        save_image(kd, path)

        a_prime = a_prime.reshape(*img_hw,-1).cpu().numpy()/(SPP//spp)
        path = dir_out['a_prime'] / '{:0>5d}_a_prime.exr'.format(i)
        imageio.imwrite(path, a_prime)
        path = dir_out['a_prime'] / '{:0>5d}_a_prime.png'.format(i)
        save_image(a_prime, path)

        roughness = roughness.reshape(*img_hw).cpu().numpy()/(SPP//spp)
        path = dir_out['roughness'] / '{:0>5d}_roughness.exr'.format(i)
        imageio.imwrite(path, roughness)
        path = dir_out['roughness'] / '{:0>5d}_roughness_color.png'.format(i)
        save_image(roughness, path, colormap=True)
        roughness = roughness[..., np.newaxis].repeat(3, -1)
        path = dir_out['roughness'] / '{:0>5d}_roughness.png'.format(i)
        save_image(roughness, path)

        metallic = metallic.reshape(*img_hw).cpu().numpy()/(SPP//spp)
        path = dir_out['metallic'] / '{:0>5d}_metallic.exr'.format(i)
        imageio.imwrite(path, metallic)
        path = dir_out['metallic'] / '{:0>5d}_metallic_color.png'.format(i)
        save_image(metallic, path, colormap=True)
        metallic = metallic[..., np.newaxis].repeat(3, -1)
        path = dir_out['metallic'] / '{:0>5d}_metallic.png'.format(i)
        save_image(metallic, path)
        
        emission = emission.reshape(*img_hw,-1).cpu().numpy()/(SPP//spp)
        path = dir_out['emission'] / '{:0>5d}_emission.exr'.format(i)
        imageio.imwrite(path, emission)
        path = dir_out['emission'] / '{:0>5d}_emission.png'.format(i)
        save_image(emission, path)
        
        merge = np.concatenate([L_gt, L_ldr, kd, a_prime, roughness, metallic, emission], axis=1)
        path = dir_out['merge'] / '{:0>5d}_merge.png'.format(i)
        save_image(merge, path)

    print('Mean PSNR: {:.5f}'.format(np.mean(psnr_list)))
    print('Mean SSIM: {:.5f}'.format(np.mean(ssim_list)))
    with open(os.path.join(dir_out['rgb'], 'metrics.txt'), 'w') as file:
        line = 'Name, PSNR, SSIM\n'
        file.write(line)
        for i in range(len(psnr_list)):
            line = '{:0>5d}, {:.5f}, {:.5f}\n'.format(i, psnr_list[i], ssim_list[i])
            file.write(line)
        line = '{:<5}, {:.5f}, {:.5f}\n'.format('mean', np.mean(psnr_list), np.mean(ssim_list))
        file.write(line)

    crfs_gt = dataset.crfs
    crfs_pred = model_crf.get_crf()
    path = os.path.join(dir_out['rgb'], 'crfs.png')
    plot_crfs(crfs_pred, crfs_gt, path)

if __name__ == '__main__':
    main()
