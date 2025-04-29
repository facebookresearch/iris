# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python -m utils.export \
    --mesh /hdd/datasets/scannetpp/data/45b0dac5e3/scans/scene.ply \
    --ckpt /hdd/meta/meta_iris_dev/checkpoints/240506_scannetpp_bathroom2/last_1.ckpt \
    --emitter_path /hdd/meta/meta_iris_dev/outputs/240506_scannetpp_bathroom2/bake \
    --dir_save /hdd/meta/meta_iris_dev/outputs/240506_scannetpp_bathroom2/output/textured_mesh \