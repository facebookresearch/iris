# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch 
import numpy as np
from matplotlib import pyplot as plt

def plot_crfs(crfs_pred, crfs_gt, path):
    crfs_pred = crfs_pred.detach().cpu().numpy()
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))
    x = np.linspace(0, 1, crfs_gt.shape[1])
    ax0.title.set_text('CRF (R)')
    ax0.set_ylabel('Pixel intensity')
    ax0.set_xlabel('Irradiance')
    ax0.plot(x, crfs_gt[0], c='b')
    ax0.plot(x, crfs_pred[0], c='r')
    ax1.title.set_text('CRF (G)')
    ax1.set_xlabel('Irradiance')
    ax1.plot(x, crfs_gt[1], c='b')
    ax1.plot(x, crfs_pred[1], c='r')
    ax2.title.set_text('CRF (B)')
    ax2.set_xlabel('Irradiance')
    ax2.plot(x, crfs_gt[2], c='b', label='GT')
    ax2.plot(x, crfs_pred[2], c='r', label='pred')
    ax2.legend()
    plt.savefig(path)
    plt.close()

def plot_weights(weights_pred, weights_gt, path):
    weights_pred = weights_pred.detach().cpu().numpy()
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))
    x = np.linspace(0, 1, weights_gt.shape[1])
    interval = 0.05
    ax0.title.set_text('Basis coefficients (R)')
    ax0.set_ylabel('value')
    ax0.bar(x-interval, weights_gt[0], interval*2)
    ax0.bar(x+interval, weights_pred[0], interval*2)
    ax1.title.set_text('Basis coefficients (G)')
    ax1.bar(x-interval, weights_gt[1], interval*2)
    ax1.bar(x+interval, weights_pred[1], interval*2)
    ax2.title.set_text('Basis coefficients (B)')
    ax2.bar(x-interval, weights_gt[2], interval*2, label='GT')
    ax2.bar(x+interval, weights_pred[2], interval*2, label='pred')
    ax2.legend()
    plt.savefig(path)
    plt.close()
