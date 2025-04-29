# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
import cv2
import os
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max_width', type=int, default=1024)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    img_names = sorted([name for name in os.listdir(args.input)])

    for name in tqdm(img_names):
        img = cv2.imread(os.path.join(args.input, name))
        if img.shape[1] > args.max_width:
            w_new = args.max_width
            h_new = int(args.max_width / img.shape[1] * img.shape[0])
            img = cv2.resize(img, (w_new, h_new))
        cv2.imwrite(os.path.join(args.output, name), img)

if __name__ == '__main__':
    main()