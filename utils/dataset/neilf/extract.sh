# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# data folder
DATASET_ROOT='/hdd/datasets/scannetpp/'
DATASET='scannetpp'
# scene name
SCENE='45b0dac5e3'
OUTPUT='/hdd/datasets/neilf/scannetpp/bathroom2/'
INPUT=ldr
LDR_IMG_DIR=Image
SPLIT=all
RES_SCALE=0.25

 python -m utils.dataset.neilf.extract_geometry \
    --dataset_root $DATASET_ROOT --scene $SCENE --dataset $DATASET \
    --input $INPUT --split $SPLIT \
    --output $OUTPUT --res_scale $RES_SCALE