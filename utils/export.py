# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import imageio
import numpy as np
import torch
import open3d as o3d
import trimesh
import xatlas
import nvdiffrast.torch as dr
from PIL import Image
from argparse import ArgumentParser
from model.brdf import NGPBRDF

def main():
    parser = ArgumentParser()
    parser.add_argument('--mesh')
    parser.add_argument('--ckpt')
    parser.add_argument('--emitter_path')
    parser.add_argument('--dir_save')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tex_res', type=int, default=2048)
    parser.add_argument('--chunk_size', type=int, default=160000)
    args = parser.parse_args()

    dir_save = args.dir_save
    os.makedirs(dir_save, exist_ok=True)

    # Load material network
    device = torch.device(args.device)
    mask = torch.load(os.path.join(args.emitter_path,'vslf.npz'),map_location='cpu')
    material_net = NGPBRDF(mask['voxel_min'],mask['voxel_max'])
    state_dict = torch.load(args.ckpt, map_location='cpu')['state_dict']
    weight = {}
    for k,v in state_dict.items():
        if 'material.' in k:
            weight[k.replace('material.','')]=v
    material_net.load_state_dict(weight)
    material_net.to(device)
    print(f'[INFO] loaded material network from {args.ckpt}')

    mesh  = trimesh.load(args.mesh)
    v_np = np.array(mesh.vertices).astype(np.float32)
    f_np = np.array(mesh.faces).astype(np.int32)

    # unwrap uvs
    print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')
    atlas = xatlas.Atlas()
    atlas.add_mesh(v_np, f_np)

    # xatlas to unwarp
    path_ft_np = os.path.join(dir_save, 'ft.npy')
    path_vt_np = os.path.join(dir_save, 'vt.npy')
    if os.path.exists(path_ft_np) and os.path.exists(path_vt_np):
        print(f'[INFO] found existing UVs, loading from {path_ft_np} and {path_vt_np}')
        ft_np = np.load(path_ft_np)
        vt_np = np.load(path_vt_np)
    else:
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 0  # disable merge_chart for faster unwrap...
        pack_options = xatlas.PackOptions()
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        _, ft_np, vt_np = atlas[0] # [N], [M, 3], [N, 2]
        np.save(path_ft_np, ft_np)
        np.save(path_vt_np, vt_np)
    print(f'[INFO] finished: xatlas unwraps UVs for mesh: v={v_np.shape} f={f_np.shape} vt={vt_np.shape} ft={ft_np.shape}')
    
    vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
    ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)
    # padding
    uv = vt * 2.0 - 1.0 # uvs to range [-1, 1]
    uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

    glctx = dr.RasterizeGLContext(output_db=False)
    tex_res = args.tex_res
    h, w = tex_res, tex_res
    # rasterize 2d texture vertices to texture image
    rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]rast
    # interpolate to get the corresponding 3D location of each pixel
    v = torch.from_numpy(v_np).to(device)
    f = torch.from_numpy(f_np).to(device)
    xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, h, w, 3]
    mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f) # [1, h, w, 1]
    vt = vt.cpu()
    ft = ft.cpu()
    uv = uv.cpu()
    # masked query
    xyzs = xyzs.view(-1, 3).cpu()
    mask = (mask > 0).view(-1).cpu()
        
    # masked query
    xyzs = xyzs.view(-1, 3).cpu()
    mask = (mask > 0).view(-1).cpu()

    mat_dim = 5
    feats = torch.zeros(h * w, mat_dim, dtype=torch.float32).cpu()
    if mask.any():
        with torch.no_grad():
            xyzs = xyzs[mask] # [M, 3]
            chunk_size = args.chunk_size
            # batched inference to avoid OOM
            all_feats = torch.zeros((xyzs.shape[0], mat_dim)).cpu()
            head = 0
            while head < xyzs.shape[0]:
                tail = min(head + chunk_size, xyzs.shape[0])
                with torch.cuda.amp.autocast(enabled=False):
                    slice_xyzs = xyzs[head:tail].clone().detach().cuda()
                    pred = material_net(slice_xyzs)
                    slice_mats = torch.cat([pred['albedo'], pred['roughness'], pred['metallic']], dim=-1)
                    all_feats[head:tail] = slice_mats.cpu().float()
                    slice_xyzs = slice_xyzs.cpu()
                    del slice_xyzs
                head += chunk_size
            feats[mask] = all_feats

    feats = feats.view(h, w, -1)
    mask = mask.view(h, w)
    # quantize [0.0, 1.0] to [0, 255]
    feats = feats.cpu().numpy()
    feats = (feats * 255).astype(np.uint8)
    mask = mask.cpu().numpy()

    albedo = np.zeros((h, w, 3), dtype=np.uint8)
    roughness_metallic = np.zeros((h, w, 3), dtype=np.uint8)
    albedo[:, :, :3] = feats[:, :, :3]
    roughness_metallic[:, :, :2] = feats[:, :, 3:]

    img_albedo = Image.fromarray(albedo)
    path_albedo = os.path.join(dir_save, 'albedo.png')
    img_albedo.save(path_albedo)
    img_roughness_metallic = Image.fromarray(roughness_metallic)
    path_roughness_metallic = os.path.join(dir_save, 'rm.png')
    img_roughness_metallic.save(path_roughness_metallic)
    print(f'[INFO] saved albedo to {path_albedo}, saved roughness and metallic to {path_roughness_metallic}')

if __name__ == '__main__':
    main()