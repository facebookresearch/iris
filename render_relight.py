# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import torch
import torch.nn.functional as NF
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import imageio

from pathlib import Path
from configs.config import default_options
from utils.dataset import RealDatasetLDR,SyntheticDatasetLDR
from utils.dataset.scannetpp.dataset import Scannetpp
from utils.ops import *
from model.brdf import NGPBRDF
from model.emitter import SLFEmitter
from model.fipt_bsdf import FIPTBSDF
from crf.model_crf import EmorCRF
from utils.disco_ball import make_disco_ball

from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
import copy
from argparse import Namespace, ArgumentParser

# disable for customized BSDF to work
import drjit as dr
dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.LoopRecord, False)
from const import GAMMA, SEED, set_random_seed
set_random_seed()

def save_image(image, path, colormap=False):
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    image = np.clip(image, 0.0, 1.0)
    image = (image*255).astype(np.uint8)
    if colormap:
        image = cv2.applyColorMap(image, cv2.COLORMAP_MAGMA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    image = image[:h-h%2, :w-w%2]
    image = Image.fromarray(image)
    image.save(path)
    return np.array(image)

def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            if(args['type'] == bool):
                parser.add_argument('--{}'.format(name), type=eval, choices=[True, False], default=str(args.get('default')))
            else:
                parser.add_argument('--{}'.format(name), **args)
        return parser

def get_mitsuba_transforms(trans_cfg):
    transform = mitsuba.ScalarTransform4f
    for trans in trans_cfg:
        if trans['type'] == 'translate':
            transform = transform.translate(trans['value'])
        elif trans['type'] == 'scale':
            transform = transform.scale(trans['value'])
        elif trans['type'] == 'rotate':
            transform = transform.rotate(axis=trans['axis'], angle=trans['angle'])

    return transform

def load_scene_dict(
        light_cfg_path,
        fov, img_hw, max_depth, 
        mesh_type, mesh_path, 
        emitter_path, brdf_path, 
    ):
    cfg = OmegaConf.load(light_cfg_path)
    cfg.PerspectiveCamera.fov = fov
    height, width = img_hw
    cfg.PerspectiveCamera.film.height = height
    cfg.PerspectiveCamera.film.width = width
    cfg.Integrator.max_depth = max_depth

    cfg.main_scene.type = mesh_type
    cfg.main_scene.filename = mesh_path 
    cfg.main_scene.bsdf.fipt_bsdf.emitter_path = emitter_path
    cfg.main_scene.bsdf.fipt_bsdf.brdf_path = brdf_path
    cfg = OmegaConf.to_container(cfg, resolve=True)

    for item_cfg in cfg.values():
        if "to_world" in item_cfg:
            item_cfg["to_world"] = get_mitsuba_transforms(item_cfg["to_world"])

    return cfg

def update_disco_from_cfg(scene_dict, disco_cfg, timestep):
    omega = 2 * np.pi / disco_cfg['T']
    make_disco_ball(
        scene_dict, 
        position=disco_cfg['position'], 
        radius=disco_cfg['radius'], 
        light_intensity=disco_cfg['light_intensity'], 
        light_num=disco_cfg['light_num'], 
        light_radius_rate=disco_cfg['light_radius_rate'],
        spot_intensity=disco_cfg['spot_intensity'], 
        spot_cutoff_angle=disco_cfg['spot_cutoff_angle'], 
        phase=timestep * omega)

def main():
    parser = ArgumentParser()
    parser = add_model_specific_args(parser)

    # add PROGRAM level args
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--mode', type=str, default='train_val', choices=['train_val', 'traj'])
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--output_path', type=str, default='outputs/kitchen_output')
    parser.add_argument('--device', type=int, required=False,default=0)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--ckpt', type=str, default='last.ckpt')
    parser.add_argument('--anti_aliasing', type=int, default=1)
    parser.add_argument('--light_cfg', type=str)
    parser.set_defaults(resume=False)
    args = parser.parse_args()
    args.gpus = [args.device]
    experiment_name = args.experiment_name
    device = torch.device(args.device)
    print('==========================')
    print('Exp:', args.experiment_name)
    print('Mode:', args.mode)
    print('Output:', args.output_path)
    print('Split:', args.split)
    print('==========================')

    dataset_name,dataset_path = args.dataset
    if dataset_name == 'synthetic':
        dataset = SyntheticDatasetLDR(
            dataset_path,
            img_dir=args.ldr_img_dir,
            split=args.split,
            pixel=False,
            ray_diff=True, 
            load_traj=True,
            res_scale=args.res_scale)
    elif dataset_name == 'real':
        dataset = RealDatasetLDR(
            dataset_path,
            img_dir=args.ldr_img_dir,
            split=args.split,
            pixel=False,
            ray_diff=True, 
            load_traj=True,
            res_scale=args.res_scale)
    elif dataset_name == 'scannetpp':
        dataset = Scannetpp(
            dataset_path, 
            args.scene, 
            split=args.split, 
            pixel=False, 
            ray_diff=True,
            load_traj=True,
            res_scale=args.res_scale)
    img_hw = dataset.img_hw

    # load geometry
    if dataset_name in ['synthetic', 'real']:
        mesh_path = os.path.join(dataset_path,'scene.obj')
        mesh_type = 'obj'
    elif dataset_name == 'scannetpp':
        mesh_path = os.path.join(dataset_path, 'data', args.scene, 'scans', 'scene.ply')
        mesh_type = 'ply'
    assert Path(mesh_path).exists(), 'mesh not found: '+mesh_path

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

    emitter_net = SLFEmitter(os.path.join(emitter_path,'emitter.pth'),
                             os.path.join(emitter_path,'vslf_0.npz'))
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

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    h_o, w_o = img_hw
    ata_factor = args.anti_aliasing
    h = h_o * ata_factor
    w = w_o * ata_factor
    img_hw = (h, w)
    # set up denoiser
    denoiser = mitsuba.OptixDenoiser([w, h])

    if dataset_name == 'synthetic':
        focal = dataset.focal
        d = w
    else:
        K = dataset.Ks[0]
        focal = K[0, 0] * ata_factor
        d = w
    fov = 2 * np.arctan(d/(focal*2))
    fov = math.degrees(fov)
    print('Fov:', fov)    

    # Set up Mitsuba config
    scene_dict = load_scene_dict(
        light_cfg_path=args.light_cfg,
        fov=fov, img_hw=(h, w), max_depth=args.indir_depth + 2,
        mesh_path=mesh_path, mesh_type=mesh_type, 
        emitter_path=emitter_path, brdf_path=str(last_ckpt),
    )
    disco_cfg = copy.deepcopy(scene_dict['disco_ball']) if 'disco_ball' in scene_dict else None

    if args.mode == 'traj':
        c2w_all = dataset.render_traj_c2w
    else:
        c2w_all = []
        for i in range(len(dataset)):
            c2w = dataset[i]['c2w'].numpy()
            c2w_all.append(c2w)
    
    if dataset_name != 'synthetic':
        c2w_all_converted = []
        for i in range(len(c2w_all)):
            c2w = c2w_all[i]
            c2w[:, :2] *= -1
            c2w_all_converted.append(c2w)
        c2w_all = c2w_all_converted

    imgs = []
    SPP = args.SPP
    spp = args.spp
    for i in tqdm(range(len(c2w_all))):
        c2w = c2w_all[i]
        t = c2w[:, 3]
        up = c2w[:, 1]
        forward = c2w[:, 2]
        
        if disco_cfg is not None:
            update_disco_from_cfg(scene_dict, disco_cfg, timestep=i)
        scene = mitsuba.load_dict(scene_dict)
        params = mitsuba.traverse(scene)
        params['PerspectiveCamera.to_world'] = mitsuba.Transform4f.look_at(origin=t, target=t+forward, up=up)
        params.update()

        img = torch.zeros(*img_hw, 3)
        seed = SEED
        for _ in range(SPP//spp):
            # render color with path tracing
            img_ = mitsuba.render(scene,spp=spp,seed=seed).torch().cpu()
            img_[img_.isnan()] = 0
            img += img_
            seed += 1

        img /= (SPP//spp)
        img = denoiser(img.numpy()).numpy()

        exposure = 1.0
        img = torch.tensor(img).reshape(-1, 3).to(device)
        img = model_crf(img, exposure)
        img = img.detach().reshape(*img_hw, -1).cpu().numpy()

        if ata_factor > 1:
            img = cv2.resize(img, (w_o, h_o), interpolation=cv2.INTER_AREA)
        path = os.path.join(output_path, '{:0>5d}_rgb.png'.format(i))
        imgs.append(save_image(img, path))
    
    if args.mode == 'traj':
        imgs += imgs[::-1]
        out_path = os.path.join(output_path, 'relight.mp4')
        imageio.mimsave(out_path, imgs, fps=30, macro_block_size=1)

if __name__ == '__main__':
    main()