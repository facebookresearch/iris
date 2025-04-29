# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# data folder
DATASET_ROOT='/hdd/datasets/scannetpp/'
DATASET='scannetpp'
# scene name
SCENE='7e09430da7'
LDR_IMG_DIR='Image'
EXP='scannetpp_room2'
VAL_FRAME=3
CRF_BASIS=3
RES_SCALE=0.5
# whether has part segmentation
HAS_PART=0
SPP=128
spp=32

# bake surface light field (SLF)
python slf_bake.py --dataset_root $DATASET_ROOT --scene $SCENE\
        --output checkpoints/$EXP/bake --res_scale $RES_SCALE\
        --dataset $DATASET 

# extract emitter mask
python extract_emitter_ldr.py \
        --dataset_root $DATASET_ROOT --scene $SCENE\
        --output checkpoints/$EXP/bake --dataset $DATASET --res_scale $RES_SCALE\
        --threshold 0.99 

python initialize.py --experiment_name $EXP --max_epochs 5 \
        --dataset $DATASET $DATASET_ROOT --scene $SCENE \
        --voxel_path checkpoints/$EXP/bake/vslf.npz \
        --emitter_path checkpoints/$EXP/bake/emitter.pth \
        --has_part $HAS_PART --val_frame $VAL_FRAME\
        --SPP $SPP --spp $spp --crf_basis $CRF_BASIS --res_scale $RES_SCALE

mv checkpoints/$EXP/last.ckpt checkpoints/$EXP/init.ckpt

# extract emitters
python extract_emitter_ldr.py --mode update\
        --dataset_root $DATASET_ROOT --scene $SCENE\
        --output checkpoints/$EXP/bake --res_scale $RES_SCALE\
        --ckpt checkpoints/$EXP/init.ckpt\
        --dataset $DATASET

python bake_shading.py \
        --dataset_root $DATASET_ROOT --scene $SCENE  \
        --dataset $DATASET --res_scale $RES_SCALE\
        --slf_path checkpoints/$EXP/bake/vslf.npz \
        --emitter_path checkpoints/$EXP/bake/emitter.pth \
        --output outputs/$EXP/shading 

# optimize BRDF, CRF
python train_brdf_crf.py --experiment_name $EXP \
        --dataset $DATASET $DATASET_ROOT --scene $SCENE\
        --has_part $HAS_PART --val_frame $VAL_FRAME --res_scale $RES_SCALE\
        --max_epochs 2 --dir_val val_0 \
        --ckpt_path checkpoints/$EXP/init.ckpt \
        --voxel_path checkpoints/$EXP/bake/vslf.npz \
        --emitter_path checkpoints/$EXP/bake/emitter.pth \
        --cache_dir outputs/$EXP/shading \
        --SPP $SPP --spp $spp --lp 0.005 --la 0.01 --l_crf_weight 0.001 --crf_basis $CRF_BASIS

mv checkpoints/$EXP/last.ckpt checkpoints/$EXP/last_0.ckpt

# refine SLF
python slf_refine.py --dataset_root $DATASET_ROOT --scene $SCENE \
        --output checkpoints/$EXP/bake --load vslf.npz --save vslf_0.npz \
        --dataset $DATASET --res_scale $RES_SCALE\
        --ckpt checkpoints/$EXP/last_0.ckpt --crf_basis $CRF_BASIS

# refine emitter 
python train_emitter.py --experiment_name $EXP \
        --dataset $DATASET $DATASET_ROOT --scene $SCENE\
        --has_part $HAS_PART --val_frame $VAL_FRAME --res_scale $RES_SCALE\
        --max_epochs 1 --dir_val val_0_emitter \
        --ckpt_path checkpoints/$EXP/last_0.ckpt \
        --voxel_path checkpoints/$EXP/bake/vslf_0.npz \
        --emitter_path checkpoints/$EXP/bake/emitter.pth \
        --SPP $SPP --spp $spp --crf_basis $CRF_BASIS

mv checkpoints/$EXP/last.ckpt checkpoints/$EXP/last_0.ckpt

# extract emitter
python extract_emitter_ldr.py --mode update\
        --dataset_root $DATASET_ROOT --scene $SCENE\
        --output checkpoints/$EXP/bake --res_scale $RES_SCALE\
        --ckpt checkpoints/$EXP/last_0.ckpt\
        --dataset $DATASET --ldr_img_dir $LDR_IMG_DIR

# refine shading 
python refine_shading.py \
        --dataset_root $DATASET_ROOT --scene $SCENE  \
        --dataset $DATASET --res_scale $RES_SCALE\
        --slf_path checkpoints/$EXP/bake/vslf_0.npz \
        --emitter_path checkpoints/$EXP/bake/emitter.pth \
        --ckpt checkpoints/$EXP/last_0.ckpt \
        --output outputs/$EXP/shading

# optimize BRDF, CRF
python train_brdf_crf.py --experiment_name $EXP \
        --dataset $DATASET $DATASET_ROOT --scene $SCENE\
        --has_part $HAS_PART --val_frame $VAL_FRAME --res_scale $RES_SCALE\
        --max_epochs 2 --dir_val val_1 \
        --ckpt_path checkpoints/$EXP/init.ckpt \
        --voxel_path checkpoints/$EXP/bake/vslf_0.npz \
        --emitter_path checkpoints/$EXP/bake/emitter.pth \
        --cache_dir outputs/$EXP/shading \
        --SPP $SPP --spp $spp --lp 0.005 --la 0.01 --l_crf_weight 0.001 --crf_basis $CRF_BASIS

mv checkpoints/$EXP/last.ckpt checkpoints/$EXP/last_1.ckpt