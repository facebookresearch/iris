# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

def compute_scale(source, target):
    '''
    solve least square error problem: target = source * scale
    '''
    source, target = source.view(-1), target.view(-1) #(n, )
    scale = torch.dot(source, target) / torch.dot(source, source)
    return scale.item()

def compute_scale_shift(source, target):
    '''
    solve least square error problem: target = source * scale + shift
    '''
    source, target = source.view(-1), target.view(-1) #(n, )
    source_one = torch.stack([source, torch.ones_like(source)], dim=-1) #(n, 2)
    pseudo_inverse = torch.inverse(source_one.T @ source_one) @ source_one.T #(2, n)
    x = pseudo_inverse @ target
    scale, shift = x[0].item(), x[1].item()
    return scale, shift

def scale_invariant_mse(source, target):
    scale = compute_scale(source, target)
    source_transformed = source * scale
    loss = F.mse_loss(source_transformed, target)
    return loss

def scale_shift_invariant_mse(source, target):
    scale, shift = compute_scale_shift(source, target)
    source_transformed = source * scale + shift
    loss = F.mse_loss(source_transformed, target)
    return loss