# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_interpolations
import os
import pathlib
from tqdm import tqdm
import scipy
from scipy import interpolate
from matplotlib import pyplot as plt
from .emor import parse_dorf_curves, parse_emor_file
from const import set_random_seed
set_random_seed()

def mono_increase_constraint(crf):
    diff = crf[1:] - crf[:-1]
    diff_min = diff.min()
    gap = -diff_min if diff_min < 0 else 0 
    diff += gap 
    diff /= diff.sum()
    crf = torch.cumsum(diff, dim=0)
    crf = torch.cat([torch.zeros((1), device=crf.device), crf])
    return crf

class EmorCRF(nn.Module):
    def __init__(self, dim=11):
        super().__init__()
        self.dim = dim
        names, vectors = parse_emor_file(inv=False)
        self.register_buffer('f0', torch.FloatTensor(vectors[1])[None])
        self.register_buffer('basis', torch.FloatTensor(vectors[2:2+dim]))
        self.weight = nn.Parameter(torch.zeros(3, dim))
    
    def get_crf(self):
        crf = self.f0 + self.weight @ self.basis
        return crf
    
    def get_inv_crf(self):
        crf = self.get_crf()
        inv_crf = []
        for i in range(3):
            crf_ch = mono_increase_constraint(crf[i])
            x = torch.linspace(0, 1, len(crf_ch)).to(self.weight.device)
            interp_func = torch_interpolations.RegularGridInterpolator([crf_ch], x)
            inv_crf_ch = interp_func([x])
            inv_crf.append(inv_crf_ch)
        inv_crf = torch.stack(inv_crf, dim=0)
        return inv_crf

    def initialize_weight(self, crf):
        weight = self.cal_weight_fitting_crf(crf) #(3, dim)
        self.weight = nn.Parameter(torch.FloatTensor(weight).to(self.weight.device))

    def cal_weight_fitting_crf(self, crf):
        f0 = self.f0.detach().cpu().numpy()
        basis = self.basis.detach().cpu().numpy().T
        pseudo_inverse = np.linalg.inv(basis.T @ basis) @ basis.T
        weight = pseudo_inverse @ (crf - f0).T 
        return weight.T
    
    def forward(self, hdr, exposure):
        '''
        Input:
            hdr: (n, 3)
        Return:
            ldr: (n, 3)
        '''
        hdr = torch.clip(hdr*exposure, 0, 1)
        crf = self.get_crf()
        x = torch.linspace(0, 1, crf.size(1)).to(self.weight.device)
        ldr = []
        for i in range(3):
            hdr_ch = hdr[:, i]
            crf_ch = crf[i]
            interp_func = torch_interpolations.RegularGridInterpolator([x], crf_ch)
            ldr_ch = interp_func([hdr_ch])
            ldr.append(ldr_ch)
        ldr = torch.stack(ldr, dim=-1)
        return ldr
    
    def inverse(self, ldr, exposure):
        '''
        Input:
            ldr: (n, 3)
        Return:
            hdr: (n, 3)
        '''
        ldr = torch.clip(ldr, 0, 1)
        inv_crf = self.get_inv_crf()
        x = torch.linspace(0, 1, inv_crf.size(1)).to(self.weight.device)
        hdr = []
        for i in range(3):
            ldr_ch = ldr[:, i]
            inv_crf_ch = inv_crf[i]
            interp_func = torch_interpolations.RegularGridInterpolator([x], inv_crf_ch)
            hdr_ch = interp_func([ldr_ch])
            hdr.append(hdr_ch)
        hdr = torch.stack(hdr, dim=-1) / exposure
        return hdr

    def reg_weight(self):
        loss = torch.mean(self.weight ** 2)
        return loss
    
    def reg_monotonically_increasing(self):
        crf = self.get_crf() #(3, 1024)
        diff = crf[:, 1:] - crf[:, :-1] # should be all positive
        loss = torch.sum(F.relu(-diff))
        return loss
    
    def reg_smoothness(self):
        crf = self.get_crf()
        smoothness = crf[:, :-2] + crf[:, 2:] - 2 * crf[:, 1:-1]
        loss = torch.mean(smoothness ** 2)
        return loss