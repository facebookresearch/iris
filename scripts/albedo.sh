# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

DIR_IMG=/hdd/datasets/scannetpp/data/4a1a3a7dc5/psdf/images/
DIR_OUTPUT=/hdd/datasets/scannetpp/data/4a1a3a7dc5/psdf/albedo/

cd /hdd/meta/IRISFormer_beta
conda activate iris
python train/train_customize.py --task_name DATE-train_mm1_albedo_eval_IIW_RESUME20230517-224958 \
    --if_train False --if_val True --if_vis True --eval_every_iter 4000 \
    --config-file train/configs/train_albedo.yaml --resume 20230517-224958--DATE-train_mm1_albedo \
    --data_root  $DIR_IMG \
    --dir_output $DIR_OUTPUT

cd /hdd/meta/meta_iris_dev
conda activate fipt
