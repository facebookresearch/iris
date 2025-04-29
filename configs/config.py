# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

default_options = {
    # dataset config
    'batch_size':{
        'type': int,
        'default': 1024*8
    },
    'dataset': {
        'type': str,
        'nargs': 2,
        'default': ['synthetic','../data/indoor_synthetic/kitchen']
    },
    'scene': {
        'type': str,
        'default': ''
    },
    'voxel_path': {
        'type': str,
        'default': 'outputs/kitchen/vslf.npz'
    },
    'num_workers': {
        'type': int,
        'default': 12
    },
    'dir_val': { # small batch of samples per point
        'type': str,
        'default': 'val'
    },
    'val_step': {
        'type': int,
        'default': 250
    },
    
    # whether has part segmentation
    'has_part': {
        'type': int,
        'default': 1
    },

    # image resolution scaling factor, 1.0 = original size
    'res_scale': {
        'type': float,
        'default': 1.0
    },
    

    # optimizer config
    'optimizer': {
        'type': str,
        'choices': ['SGD', 'Ranger', 'Adam'],
        'default': 'Adam'
    },
    'learning_rate': {
        'type': float,
        'default': 1e-3
    },
    'weight_decay': {
        'type': float,
        'default': 0
    },

    'scheduler_rate':{
        'type': float,
        'default': 0.5
    },
    'milestones':{
        'type': int,
        'nargs': '*',
        'default': [1000]
    },
    
    
    # reuglarization config
    'le': {
        'type': float,
        'default': 1.0
    },
    'ld': {
        'type': float,
        'default': 5e-4
    },
    'lp': {
        'type': float,
        'default': 5e-3
    },
    'ls': {
        'type': float,
        'default': 1e-3
    },
    'la': {
        'type': float,
        'default': 0.0
    },
    'sigma_albedo': {
        'type': float,
        'default': 0.05/3.0
    },
    'sigma_pos': {
        'type': float,
        'default': 0.3/3.0
    },

    # model params
    'ckpt_path': {
        'type': str,
        'default': None
    },
    'emitter_path': {
        'type': str,
        'default': None
    },
    'freeze_emitter': {
        'type': int,
        'default': 0
    },
    'freeze_crf': {
        'type': int,
        'default': 0
    },
    'indir_depth': {
        'type': int,
        'default': 5
    },
    'SPP': { # Total samples per point
        'type': int,
        'default': 512
    },
    'spp': { # small batch of samples per point
        'type': int,
        'default': 8
    },

    # LDR images with varying exposure
    'ldr_img_dir': {
        'type': str,
        'default': None
    },
     'crf_basis': {
        'type': int,
        'default': 3
    },
    'load_crf': {
        'type': int,
        'default': 0
    },
    'l_crf_increasing': {
        'type': float,
        'default': 0.1
    },
    'l_crf_weight': {
        'type': float,
        'default': 0.001
    },
}
