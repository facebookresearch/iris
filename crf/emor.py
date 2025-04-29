# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import pathlib
from matplotlib import pyplot as plt
from const import set_random_seed
set_random_seed()

repo_root = pathlib.Path().resolve()
emor_path = os.path.join(repo_root, 'crf', 'emor.txt')
invemor_path = os.path.join(repo_root, 'crf', 'invemor.txt')
curves_path = os.path.join(repo_root, 'crf', 'dorfCurves.txt')

def parse_emor_file(inv=True):
    file_path = invemor_path if inv else emor_path
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    
    stride = 1 + 256
    names = []
    vectors = []
    for i in range(len(lines)//stride):
        name_line = lines[i*stride]
        name = name_line.split('=')[0].strip()
        names.append(name)
        lines_numbers = lines[i*stride+1: (i+1)*stride]
        numbers = [line.split() for line in lines_numbers]
        vector = np.float32(numbers).reshape(-1)
        vectors.append(vector)
    names = np.array(names)
    vectors = np.stack(vectors)
    return names, vectors

def parse_dorf_curves():
    with open(curves_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    stride = 6
    names = []
    vectors = []
    for i in range(len(lines)//stride):
        line_sample = lines[i*stride: (i+1)*stride]
        n_i = '{}-{}-{}'.format(line_sample[0], line_sample[1], line_sample[2][0])
        n_b = '{}-{}-{}'.format(line_sample[0], line_sample[1], line_sample[4][0])
        names += [n_b]
        v_i = np.float32(line_sample[3].split())
        v_b = np.float32(line_sample[5].split())
        vectors += [v_b]
    names = np.array(names)
    vectors = np.stack(vectors)
    return names, vectors #(201, 1024)

def get_dorf_mean_basis(top=25):
    names, curves = parse_dorf_curves()
    mean = np.mean(curves, 0)
    curves = curves - mean[None]
    u, s, vh = np.linalg.svd(curves)
    scaled_basis = s[:top, None] * vh[:top]
    # scaled_basis = vh[:top]
    return mean, scaled_basis

def mono_increase_constraint(crf):
    diff = crf[1:] - crf[:-1]
    gap = -1 * np.min([0.0, diff.min()])
    diff += gap 
    diff /= diff.sum()
    crf = np.cumsum(diff)
    crf = np.concatenate([np.zeros((1)), crf])
    return crf

