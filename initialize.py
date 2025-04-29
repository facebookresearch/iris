# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
torch.set_float32_matmul_precision('high')
import torch.nn.functional as NF
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_scatter
import time
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')
import math
import numpy as np
import os
from pathlib import Path
from argparse import Namespace, ArgumentParser

from configs.config import default_options
from utils.dataset import InvRealDatasetLDR,RealDatasetLDR,InvSyntheticDatasetLDR,SyntheticDatasetLDR
from utils.dataset.scannetpp.dataset import Scannetpp, InvScannetpp
from utils.ops import *
from utils.path_tracing import ray_intersect, path_tracing, path_tracing_single
from model.mlps import ImplicitMLP
from model.brdf import NGPBRDF
from model.emitter import SLFEmitter, SLFEmitterLearn
from crf.model_crf import EmorCRF
from crf.plot import plot_crfs
from render import save_image
from const import GAMMA, set_random_seed
set_random_seed()

class ModelTrainer(pl.LightningModule):
    """ BRDF-emission mask training code """
    def __init__(self, hparams: Namespace, *args, **kwargs):
        super(ModelTrainer, self).__init__()
        self.save_hyperparameters(hparams)
        
        dataset, dataset_root = hparams.dataset
        scene = hparams.scene
        if dataset in ['synthetic', 'real']:
            mesh_path = os.path.join(dataset_root,'scene.obj')
            mesh_type = 'obj'
        elif dataset == 'scannetpp':
            mesh_path = os.path.join(dataset_root, 'data', scene, 'scans', 'scene.ply')
            mesh_type = 'ply'
        assert Path(mesh_path).exists(), 'mesh not found: '+mesh_path
        # load scene geometry
        self.scene = mitsuba.load_dict({
            'type': 'scene',
            'shape_id':{
                'type': mesh_type,
                'filename': mesh_path
            }
        })

        # initiallize BRDF
        mask = torch.load(hparams.voxel_path,map_location='cpu')
        material_net = NGPBRDF(mask['voxel_min'],mask['voxel_max'])
        if hparams.ckpt_path:
            state_dict = torch.load(hparams.ckpt_path, map_location='cpu')['state_dict']
            weight = {}
            for k,v in state_dict.items():
                if 'material.' in k:
                    weight[k.replace('material.','')]=v
            material_net.load_state_dict(weight)
            
        self.material = material_net
       
        # initialize emission mask
        emitter = SLFEmitterLearn(
            emitter_path=hparams.emitter_path,
            slf_path=hparams.voxel_path
        )
        self.emitter = emitter

        model_crf = EmorCRF(dim=hparams.crf_basis)
        for p in model_crf.parameters():
            p.requires_grad=False
        self.model_crf = model_crf
        
    def __repr__(self):
        return repr(self.hparams)

    def configure_optimizers(self):
        if(self.hparams.optimizer == 'SGD'):
            opt = optim.SGD
        if(self.hparams.optimizer == 'Adam'):
            opt = optim.Adam
        
        optimizer = opt(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)    
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.hparams.milestones,gamma=self.hparams.scheduler_rate)
        return [optimizer], [scheduler]
    
    def train_dataloader(self,):
        dataset_name,dataset_path = self.hparams.dataset
        if dataset_name == 'synthetic':
            dataset = InvSyntheticDatasetLDR(dataset_path,img_dir=hparams.ldr_img_dir,pixel=True,split='train',
                                       batch_size=self.hparams.batch_size,has_part=self.hparams.has_part)
        elif dataset_name == 'real':
            dataset = InvRealDatasetLDR(dataset_path,img_dir=hparams.ldr_img_dir,pixel=True,split='train',
                                       batch_size=self.hparams.batch_size)
        elif dataset_name == 'scannetpp':
            scene = self.hparams.scene
            dataset = InvScannetpp(dataset_path, scene, pixel=True, split='train',
                                   batch_size=self.hparams.batch_size, res_scale=hparams.res_scale)
        self.train_dataset = dataset
        self.train_loader = DataLoader(dataset, batch_size=None, num_workers=self.hparams.num_workers)
        return self.train_loader
       
    def on_train_epoch_start(self,):
        """ resample training batch """
        self.train_loader.dataset.resample()
    
    def val_dataloader(self,):
        dataset_name,dataset_path = self.hparams.dataset
        self.dataset_name = dataset_name
        if dataset_name == 'synthetic':
            dataset = SyntheticDatasetLDR(dataset_path,img_dir=hparams.ldr_img_dir,pixel=False,split='val',ray_diff=True, val_frame=self.hparams.val_frame)
        elif dataset_name == 'real':
            dataset = RealDatasetLDR(dataset_path,img_dir=hparams.ldr_img_dir,pixel=False,split='val',ray_diff=True, val_frame=self.hparams.val_frame)
        elif dataset_name == 'scannetpp':
            scene = self.hparams.scene
            dataset = Scannetpp(dataset_path, scene, pixel=False, split='test', ray_diff=True, val_frame=self.hparams.val_frame, res_scale=hparams.res_scale)
        self.img_hw = dataset.img_hw
        self.val_dataset = dataset
        self.val_loader = DataLoader(dataset, shuffle=False, batch_size=None, num_workers=self.hparams.num_workers)
        return self.val_loader

    def forward(self, points, view):
        return

    def gamma(self,x):
        """ tone mapping function """
        mask = x <= 0.0031308
        ret = torch.empty_like(x)
        ret[mask] = 12.92*x[mask]
        mask = ~mask
        ret[mask] = 1.055*x[mask].pow(1/2.4) - 0.055
        return ret
    
    def training_step(self, batch, batch_idx):
        """ one training step """
        rays,rgbs_gt = batch['rays'], batch['rgbs']
        xs,ds = rays[...,:3],rays[...,3:6]
        ds = NF.normalize(ds,dim=-1)
        dxdu,dydv = rays[...,6:9],rays[...,9:12]
        
        # find surface intersection
        du,dv = torch.rand(2,len(xs),1,device=xs.device)-0.5
        wi = NF.normalize(ds+dxdu*du+dydv*dv,dim=-1)
        positions,normals,_,triangle_idx,valid = ray_intersect(self.scene,xs,wi)

        if not valid.any():
            return None
        
        # diffuse regualrization
        mat = self.material(positions)
        albedo,metallic,roughness = mat['albedo'],mat['metallic'],mat['roughness']

        # only optimize emitter 
        for param in self.material.parameters():
            param.requires_grad = False 
        SPP = self.hparams.SPP
        spp = self.hparams.spp
        L = torch.zeros_like(xs)
        for _ in range(SPP//spp):
            L += path_tracing_single(
                self.scene, self.emitter, self.material,
                xs, ds, dxdu, dydv, spp 
            )
        L = L / (SPP//spp) 
        
        exposure = batch['exposure']
        rgbs_ldr = self.model_crf(L, exposure)
        loss_c = NF.mse_loss(rgbs_ldr, rgbs_gt)
        for param in self.material.parameters():
            param.requires_grad = True 

        segmentation = batch['segmentation'].long()
        seg_idxs,inv_idxs = segmentation.unique(return_inverse=True)

        int_albedo = batch['int_albedo']
        weight_seg = torch.zeros(len(seg_idxs),device=seg_idxs.device)
        weight_seg_ = torch.ones_like(roughness).squeeze(-1).detach()
        weight_seg = torch_scatter.scatter(weight_seg_,inv_idxs,0,weight_seg,reduce='sum').unsqueeze(-1)

        mean_albedo = torch.zeros(len(seg_idxs), 3, device=seg_idxs.device)
        mean_albedo = torch_scatter.scatter(
                    int_albedo*weight_seg_.unsqueeze(-1),inv_idxs,0,mean_albedo,reduce='sum')
        mean_albedo = mean_albedo / weight_seg
        mean_albedo = mean_albedo[inv_idxs]
        loss_a = NF.mse_loss(albedo, mean_albedo)
        loss = loss_a + loss_c

        # mask out emissive regionce
        # vsualize rendering brdf
        psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))
        if self.dataset_name == 'synthetic':
            albedos_gt = batch['albedo']
            albedo_loss = NF.mse_loss(albedos_gt,albedo)
            self.log('init/albedo', albedo_loss)
            roughness_gt = batch['roughness']
            roughness_loss = NF.mse_loss(roughness_gt, roughness.squeeze(-1))
            self.log('init/roughness', roughness_loss)
        self.log('init/loss', loss)
        self.log('init/loss_c', loss_c)
        self.log('init/loss_a', loss_a)
        self.log('init/psnr', psnr)

        val_step = self.hparams.val_step
        if self.global_step % val_step == 0: # and self.global_step > 0:
            val_frame = self.val_dataset.val_frame
            batch = self.val_dataset[val_frame]
            self.validation(batch)
            
        return loss
    
    def validation(self, batch):
        # print('[val in training]')
        SPP = self.hparams.SPP
        spp = self.hparams.spp
        img_hw = self.img_hw
        denoiser = mitsuba.OptixDenoiser(img_hw[::-1])
        dir_val = os.path.join('outputs', self.hparams.experiment_name, 'init')
        os.makedirs(dir_val, exist_ok=True)
            
        device = torch.device(0)
        rays,rgbs_gt = batch['rays'].to(device), batch['rgbs']
        rays_o, rays_d = rays[...,:3],rays[...,3:6]
        rays_d = NF.normalize(rays_d,dim=-1)
        dxdu,dydv = rays[...,6:9],rays[...,9:12]
            
        L_train = torch.zeros_like(rays_o)
        L_full  = torch.zeros_like(rays_o)
        albedo = torch.zeros_like(rays_o)
        roughness = torch.zeros_like(rays_o[..., :1])
        metallic = torch.zeros_like(rays_o[..., :1])
        emission = torch.zeros_like(rays_o)
        with torch.no_grad():
            for _ in range(SPP//spp):
                L_train += path_tracing_single(
                    self.scene, self.emitter, self.material,
                    rays_o, rays_d, dxdu, dydv, spp 
                )
                L_full  += path_tracing(
                    self.scene, self.emitter, self.material,
                    rays_o, rays_d, dxdu, dydv, spp,
                    indir_depth=5
                )

                # sample pixels
                du,dv = torch.rand(2,len(rays_o),spp,1,device=device)
                ds = rays_d[:,None]+ dxdu[:,None]*du + dydv[:,None]*dv
                ds = NF.normalize(ds,dim=-1).reshape(-1,3)
                xs = rays_o.repeat_interleave(spp,dim=0)

                positions,normals,_,triagnle_idxs,valid = ray_intersect(self.scene,xs,ds)

                mat = self.material(positions)

                # get brdf parameters
                albedo_ = mat['albedo']
                metallic_ = mat['metallic']
                roughness_ = mat['roughness']

                # find emission
                emission_ = self.emitter.eval_emitter(positions,ds,triagnle_idxs)[0]
                emit_mask = emission_.sum(-1,keepdim=True)==0
                valid = valid.unsqueeze(-1)

                # scene intrinsics
                albedo += (albedo_*valid*emit_mask).reshape(-1,spp,3).mean(1)
                roughness += (roughness_*valid*emit_mask).reshape(-1,spp,1).mean(1)
                metallic += (metallic_*valid*emit_mask).reshape(-1,spp,1).mean(1)
                emission += emission_.reshape(-1,spp,3).mean(1)


        L_train = L_train / (SPP//spp)
        L_train = L_train.reshape(*img_hw, -1).cpu()
        L_train = denoiser(L_train.numpy()).numpy()
    
        exposure = batch['exposure']
        L_train = torch.tensor(L_train).reshape(-1, 3).to(device)
        L_train = self.model_crf(L_train, exposure)
        L_train = L_train.detach().reshape(*img_hw, -1).cpu().numpy()

        path = os.path.join(dir_val, '{:0>5d}_L_train.png'.format(self.global_step))
        save_image(L_train, path)
            
        L_full  = L_full  / (SPP//spp)
        L_full = L_full.reshape(*img_hw, -1).cpu()
        L_full = denoiser(L_full.numpy()).numpy()
        
        exposure = batch['exposure']
        L_full = torch.tensor(L_full).reshape(-1, 3).to(device)
        L_full = self.model_crf(L_full, exposure)
        L_full = L_full.detach().reshape(*img_hw, -1).cpu().numpy()

        path = os.path.join(dir_val, '{:0>5d}_L_full.png'.format(self.global_step))
        save_image(L_full, path)

        L_gt = rgbs_gt.reshape(*img_hw, -1).cpu().numpy()
        path = os.path.join(dir_val, '{:0>5d}_L_gt.png'.format(self.global_step))
        save_image(L_gt, path)

        albedo = albedo.reshape(*img_hw,-1).cpu()/(SPP//spp)
        path = os.path.join(dir_val, '{:0>5d}_mat_albedo.png'.format(self.global_step))
        save_image(albedo, path)
        roughness = roughness.reshape(*img_hw,1).cpu()/(SPP//spp)
        path = os.path.join(dir_val, '{:0>5d}_mat_roughness.png'.format(self.global_step))
        save_image(roughness, path, colormap=True)
        metallic = metallic.reshape(*img_hw,1).cpu()/(SPP//spp)
        path = os.path.join(dir_val, '{:0>5d}_mat_metallic.png'.format(self.global_step))
        save_image(metallic, path, colormap=True)
        emission = emission.reshape(*img_hw,-1).cpu()/(SPP//spp)/20
        path = os.path.join(dir_val, '{:0>5d}_emission.png'.format(self.global_step))
        save_image(emission, path)
        crfs_gt = self.val_dataset.crfs
        crfs_pred = self.model_crf.get_crf()
        path = os.path.join(dir_val, '{:0>5d}_crfs.png'.format(self.global_step))
        plot_crfs(crfs_pred, crfs_gt, path)

    
    def validation_step(self, batch, batch_idx):
        """ visualize diffuse reflectance kd
        """
        rays,rgb_gt = batch['rays'], batch['rgbs']
        if self.dataset_name == 'synthetic':
            emission_mask_gt = batch['emission'].mean(-1,keepdim=True) == 0
        else:
            emission_mask_gt = torch.ones_like(rays[...,:1])
        rays_x = rays[:,:3]
        rays_d = NF.normalize(rays[:,3:6],dim=-1)

        positions,normals,_,_,valid = ray_intersect(self.scene,rays_x,rays_d)
        position = positions[valid]

        # batched rendering diffuse reflectance
        B = valid.sum()
        batch_size = 10240
        albedo_ = []
        for b in range(math.ceil(B*1.0/batch_size)):
            b0 = b*batch_size
            b1 = min(b0+batch_size,B)
            mat = self.material(position[b0:b1])
            albedo_.append(mat['albedo']*(1-mat['metallic']))
        albedo_ = torch.cat(albedo_)
        albedo = torch.zeros(len(valid),3,device=valid.device)
        albedo[valid] = albedo_
        
        if self.dataset_name == 'synthetic':
            albedo_gt = batch['albedo']
        else: # show rgb is no ground truth kd
            albedo_gt = rgb_gt.pow(1/GAMMA).clamp(0,1)

        # mask out emissive regions
        albedo = albedo*emission_mask_gt
        albedo_gt = albedo_gt * emission_mask_gt
        loss_c = NF.mse_loss(albedo_gt,albedo)
        
        loss = loss_c
        psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))
        
        
        self.log('val/loss', loss)
        self.log('val/psnr', psnr)
        return

            
def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            if(args['type'] == bool):
                parser.add_argument('--{}'.format(name), type=eval, choices=[True, False], default=str(args.get('default')))
            else:
                parser.add_argument('--{}'.format(name), **args)
        return parser
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_model_specific_args(parser)
    hparams, _ = parser.parse_known_args()

    # add PROGRAM level args
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--ft', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--device', type=int, required=False,default=None)
    parser.add_argument('--val_frame', type=int, default=0)

    parser.set_defaults(resume=False)
    args = parser.parse_args()
    args.gpus = [args.device]
    hparams.experiment_name = args.experiment_name
    hparams.val_frame = args.val_frame
    experiment_name = args.experiment_name

    # setup checkpoint loading
    checkpoint_path = Path(args.checkpoint_path) / experiment_name
    log_path = Path(args.log_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val/loss', save_top_k=1, save_last=True)

    last_ckpt = checkpoint_path / 'last.ckpt' if args.resume else None
    if (last_ckpt is None) or (not (last_ckpt.exists())):
        last_ckpt = None
    else:
        last_ckpt = str(last_ckpt)
    
    # setup model trainer
    model = ModelTrainer(hparams)
    
    
    # Update to lightning 1.9
    trainer = Trainer.from_argparse_args(
        args,
        accelerator='gpu', devices=[0], gpus=None, 
        # logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        max_epochs=args.max_epochs, 
    )

    start_time = time.time()
    
    trainer.fit(
        model, 
        ckpt_path=last_ckpt, 
        )
    
    print('[train - BRDF-emission] time (s): ', time.time()-start_time)
