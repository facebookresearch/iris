# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

DATA_ROOT=/hdd/datasets/scannetpp/data
MAX_W=1024

scenes=(
    '1ada7a0617'
)

for scene in "${scenes[@]}"; do
    echo "============== $scene =============="
    cp $DATA_ROOT/$scene/dslr/train_test_lists.json $DATA_ROOT/$scene/psdf

    # Extract semantic
    python -m utils.dataset.scannetpp.gen_mesh        --scene $scene
    python -m utils.dataset.scannetpp.render_semantic --scene $scene

    # Estimate albedo
    cd /hdd/meta/IRISFormer_beta
    conda activate iris
    python train/train_customize.py --task_name DATE-train_mm1_albedo_eval_IIW_RESUME20230517-224958 \
        --if_train False --if_val True --if_vis True --eval_every_iter 4000 \
        --config-file train/configs/train_albedo.yaml --resume 20230517-224958--DATE-train_mm1_albedo \
        --data_root  $DATA_ROOT/$scene/dslr/undistorted_images \
        --dir_output $DATA_ROOT/$scene/dslr/albedo

    cd /hdd/meta/meta_iris_dev
    conda activate fipt

    # resize albedo
    python -m utils.dataset.scannetpp.process \
        --input  $DATA_ROOT/$scene/dslr/albedo/ \
        --output $DATA_ROOT/$scene/psdf/albedo/ \
        --max_width $MAX_W

    # resize semantic mask
    python -m utils.dataset.scannetpp.process \
        --input  $DATA_ROOT/$scene/dslr/seg/ \
        --output $DATA_ROOT/$scene/psdf/seg/ \
        --max_width $MAX_W
done

