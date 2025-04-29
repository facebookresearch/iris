# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


python -m utils.metric_crf \
    --crf_gt /hdd/datasets/fipt/indoor_synthetic/livingroom/train/Image/cam/crf.npy \
    --ckpt checkpoints/250127_no_crf_livingroom/last_1.ckpt \
    --dir_save outputs/250127_no_crf_livingroom/