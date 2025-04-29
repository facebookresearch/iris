# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
from scipy import interpolate
from .emor import parse_dorf_curves, parse_emor_file

def open_exr(file):
    img = cv2.imread(file,cv2.IMREAD_UNCHANGED)[...,[2,1,0]]
    # img = img.astype(np.float32)
    return img

def apply_crf(image, curves):
    h, w, _ = image.shape
    image_ldr = []
    for i in range(3):
        ch = image[:, :, i].reshape(-1)
        crf = curves[i]
        x = np.linspace(0, 1, len(crf))
        crf_inter = interpolate.interp1d(x, crf)
        ch_ldr = crf_inter(ch).reshape(h, w)
        image_ldr.append(ch_ldr)
    image_ldr = np.stack(image_ldr, axis=-1)
    return image_ldr

def main():
    parser = ArgumentParser()
    parser.add_argument('--dir_src', help='dir path to HDR images')
    parser.add_argument('--dir_tgt', help='dir path to output LDR images')
    parser.add_argument('--curve_idx', nargs='+', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.dir_tgt, exist_ok=True)
    
    level = np.array([-2, -1, 0, 1, 2])
    exposure = (np.ones(len(level))*2)**level

    names_all, curves_all = parse_dorf_curves()
    idx = args.curve_idx
    names = names_all[[idx[0], idx[1], idx[2]]]
    curves = curves_all[[idx[0], idx[1], idx[2]]]
    dir_cam = os.path.join(args.dir_tgt, 'cam')
    os.makedirs(dir_cam, exist_ok=True)
    for i in range(len(names)):
        n = names[i]
        crf = curves[i]
        x = np.linspace(0, 1, len(crf))
        plt.title(n)
        plt.plot(x, crf)
        channel = 'R' if i==0 else 'G' if i==1 else 'B'
        out_path = os.path.join(dir_cam, 'crf_{}.png'.format(channel))
        plt.savefig(out_path)
        plt.close()
    np.save(os.path.join(dir_cam, 'crf.npy'), curves)

    save_sorted_exposure(args.dir_src, args.dir_tgt, exposure, curves)

def save_all_exposure(dir_src, dir_tgt, exposure, curves):
    hdr_paths = sorted([os.path.join(dir_src, name) for name in os.listdir(dir_src) if name.endswith('.exr')])
    for path in tqdm(hdr_paths):
        hdr = open_exr(path)
        prefix = path.split('/')[-1].split('.')[0]
        for i, exp in enumerate(exposure):
            irr = hdr * exp
            irr = np.clip(irr, 0, 1)
            ldr = apply_crf(irr, curves)
            out_path = os.path.join(dir_tgt, '{}_{}.png'.format(prefix, i))
            img = Image.fromarray((ldr*255).astype(np.uint8))
            img.save(out_path)

def save_sorted_exposure(dir_src, dir_tgt, exposure, curves):
    hdr_paths = sorted([os.path.join(dir_src, name) for name in os.listdir(dir_src) if name.endswith('.exr')])
    img_means = []
    for path in hdr_paths:
        img = open_exr(path)
        mean = np.mean(img)
        img_means.append(mean)
    img_means = np.array(img_means)

    argsort = np.argsort(img_means)
    exposure = np.sort(exposure)[::-1] #big to small
    img_exp = np.zeros_like(img_means)
    step = len(img_exp) // len(exposure)
    for i, exp in enumerate(exposure):
        idx = argsort[i*step:(i+1)*step]
        img_exp[idx] = exp
    idx = argsort[step*len(exposure):]
    img_exp[idx] = exposure[-1]
    np.save(os.path.join(dir_tgt, 'cam', 'exposure.npy'), img_exp)

    for i in tqdm(range(len(hdr_paths))):
        path = hdr_paths[i]
        hdr = open_exr(path)
        exp = img_exp[i]
        irr = np.clip(hdr * exp, 0, 1)
        ldr = apply_crf(irr, curves)
        prefix = path.split('/')[-1].split('.')[0]
        ldr_path = os.path.join(dir_tgt, '{}.png'.format(prefix))
        img = Image.fromarray((ldr*255).astype(np.uint8))
        img.save(ldr_path)

if __name__ == '__main__':
    main()
