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
from crf.model_crf import EmorCRF
import plotly.graph_objects as go
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--crf_gt', help='path to LDR_IMG_DIR/cam/crf.npy')
parser.add_argument('--ckpt', help='path to checkpoint last_1.ckpt')
parser.add_argument('--dir_save', help='dir for saving image')
args = parser.parse_args()

# SCENE = 'kitchen'
# METHOD = '1104_kitchen_single_base'
# LDR_IMG_DIR='Image'
# GT_PATH = '/hdd/datasets/fipt/indoor_synthetic/'
# path_gt_crf = os.path.join(GT_PATH, SCENE, 'train', LDR_IMG_DIR, 'cam', 'crf.npy')
crf_gt = np.load(args.crf_gt)
last_ckpt = os.path.join(args.ckpt)

n_basis = 3
model_crf = EmorCRF(dim=n_basis)
state_dict = torch.load(last_ckpt, map_location='cpu')['state_dict']
weight = {}
for k,v in state_dict.items():
    if 'model_crf.' in k:
        weight[k.replace('model_crf.','')]=v
model_crf.load_state_dict(weight)
crf_pred = model_crf.get_crf().detach().numpy()

l2 = np.sqrt(np.sum((crf_gt - crf_pred)**2))
l2 = np.linalg.norm(crf_gt - crf_pred)
print('L2: {:.5f}'.format(l2))

def plot_plotly():
    x = np.linspace(0, 1, 1024)
    y1 = crf_gt[0]
    y2 = crf_pred[0]
    trace1 = go.Scatter(x=x, y=y1, mode='lines', name='GT', line=dict(color='blue', width=2, dash='dash'))
    trace2 = go.Scatter(x=x, y=y2, mode='lines', name='Pred.', line=dict(color='red', width=2))

    # Create layout
    layout = go.Layout(title='', xaxis=dict(title='Irradiance', range=[0, 1]), yaxis=dict(title='LDR', range=[0, 1]))

    # Create a figure
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Show the plot
    fig.show()

def plot_test_dark():
    x = np.linspace(0, 1, 1024)
    line = crf_pred[0]
    
    fig, ax = plt.subplots()
    ax.set_facecolor('black')
    ax.plot(x, line, color='yellow')
    # ax.xaxis.label.set_color('white')
    # ax.yaxis.label.set_color('white')
    ax.set_xlabel('Irradiance', size=18)
    ax.set_ylabel('Pixel Value', size=18)
    # ax.tick_params(axis='x', labelsize=12)
    # ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(which='both', bottom=False, left=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color='white', linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')
    plt.show()

def plot_test():
    x = np.linspace(0, 1, 1024)
    line = crf_pred[0]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor((0.9, 0.9, 0.9))
    ax.plot(x, line, color=(0, 0, 1), linestyle='--', linewidth=2.5)
    # ax.xaxis.label.set_color('white')
    # ax.yaxis.label.set_color('white')
    ax.set_xlabel('Irradiance', size=24)
    ax.set_ylabel('Pixel Value', size=24)
    # ax.tick_params(axis='x', labelsize=12)
    # ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(which='both', bottom=False, left=False, labelsize=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color=(0, 0, 0), linestyle='--', linewidth=0.5)
    ax.set_aspect('equal')
    plt.show()

def plot_single():
    x = np.linspace(0, 1, 1024)
 
    fig, ax = plt.subplots(figsize=(10, 10))
    # ax.set_facecolor((0.95, 0.95, 0.95))
    ax.plot(x, crf_pred[0], color=(0, 0.1882, 0.6), linewidth=8.0, label='Pred.')
    ax.plot(x, crf_gt[0], color=(0.97, 0.5, 0), linestyle='--', linewidth=8.0, label='GT')
    ax.legend(fontsize=32)
    # ax.xaxis.label.set_color('white')
    # ax.yaxis.label.set_color('white')
    ax.set_xlabel('Irradiance', size=48)
    ax.set_ylabel('Pixel Value', size=48)
    # ax.tick_params(axis='x', labelsize=12)
    # ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(which='both', bottom=False, left=False, labelsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color=(0, 0, 0), linestyle='--', linewidth=1.0)
    ax.set_aspect('equal')
    path = os.path.join(args.dir_save, 'crf.png')
    plt.savefig(path)
    print('Saved to', path)

def plot_all():
    x = np.linspace(0, 1, 1024)
 
    fig, ax = plt.subplots(figsize=(10, 10))
    # ax.set_facecolor((0.95, 0.95, 0.95))
    ax.plot(x, crf_pred[0], color=(1, 0, 0), linewidth=4.0, label='R Pred.')
    ax.plot(x, crf_gt[0],   color=(1, 0, 0), linewidth=4.0, label='R GT', linestyle='--')
    ax.plot(x, crf_pred[1], color=(0, 1, 0), linewidth=4.0, label='G Pred.')
    ax.plot(x, crf_gt[1],   color=(0, 1, 0), linewidth=4.0, label='G GT', linestyle='--')
    ax.plot(x, crf_pred[2], color=(0, 0, 1), linewidth=4.0, label='B Pred.')
    ax.plot(x, crf_gt[2],   color=(0, 0, 1), linewidth=4.0, label='B GT', linestyle='--')
    ax.legend(fontsize=32)
    # ax.xaxis.label.set_color('white')
    # ax.yaxis.label.set_color('white')
    ax.set_xlabel('Irradiance', size=48)
    ax.set_ylabel('Pixel Value', size=48)
    # ax.tick_params(axis='x', labelsize=12)
    # ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(which='both', bottom=False, left=False, labelsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color=(0, 0, 0), linestyle='--', linewidth=1.0)
    ax.set_aspect('equal')
    path = os.path.join(args.dir_save, 'crf_all.png')
    plt.savefig(path)
    print('Saved to', path)

# plot_single()
# plot_all()