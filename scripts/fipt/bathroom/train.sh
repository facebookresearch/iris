# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# data folder
DATASET_ROOT='/hdd/datasets/fipt/indoor_synthetic/'
DATASET='synthetic'
# scene name
SCENE='bathroom'
LDR_IMG_DIR='Image'
EXP='fipt_syn_bathroom'
VAL_FRAME=10 
CRF_BASIS=3
# whether has part segmentation
HAS_PART=1
SPP=128
spp=32

# bake surface light field (SLF)
python slf_bake.py --scene $DATASET_ROOT$SCENE\
        --output checkpoints/$EXP/bake \
        --dataset $DATASET --ldr_img_dir $LDR_IMG_DIR

# extract emitter mask
python extract_emitter_ldr.py --scene $DATASET_ROOT$SCENE\
        --output checkpoints/$EXP/bake --dataset $DATASET\
        --ldr_img_dir $LDR_IMG_DIR --threshold 0.99 

python initialize.py --experiment_name $EXP --max_epochs 6 \
        --dataset $DATASET $DATASET_ROOT$SCENE \
        --voxel_path checkpoints/$EXP/bake/vslf.npz \
        --emitter_path checkpoints/$EXP/bake/emitter.pth \
        --has_part $HAS_PART --ldr_img_dir $LDR_IMG_DIR  --val_frame $VAL_FRAME\
        --SPP $SPP --spp $spp --crf_basis $CRF_BASIS

mv checkpoints/$EXP/last.ckpt checkpoints/$EXP/init.ckpt

# extract emitters
python extract_emitter_ldr.py --mode update\
        --scene $DATASET_ROOT$SCENE\
        --output checkpoints/$EXP/bake\
        --ckpt checkpoints/$EXP/init.ckpt\
        --dataset $DATASET --ldr_img_dir $LDR_IMG_DIR

python bake_shading.py \
        --scene $DATASET_ROOT$SCENE --dataset $DATASET \
        --ldr_img_dir $LDR_IMG_DIR \
        --slf_path checkpoints/$EXP/bake/vslf.npz \
        --emitter_path checkpoints/$EXP/bake/emitter.pth \
        --output outputs/$EXP/shading 

# optimize BRDF, CRF
python train_brdf_crf.py --experiment_name $EXP \
        --max_epochs 4 --dir_val val_0 \
        --ckpt_path checkpoints/$EXP/init.ckpt \
        --voxel_path checkpoints/$EXP/bake/vslf.npz \
        --emitter_path checkpoints/$EXP/bake/emitter.pth \
        --cache_dir outputs/$EXP/shading \
        --dataset $DATASET $DATASET_ROOT$SCENE \
        --has_part $HAS_PART --ldr_img_dir $LDR_IMG_DIR  --val_frame $VAL_FRAME\
        --SPP $SPP --spp $spp --lp 0.005 --la 0.01 --l_crf_weight 0.001 --crf_basis $CRF_BASIS

mv checkpoints/$EXP/last.ckpt checkpoints/$EXP/last_0.ckpt

# refine SLF
python slf_refine.py --scene $DATASET_ROOT$SCENE \
        --output checkpoints/$EXP/bake --load vslf.npz --save vslf_0.npz \
        --dataset $DATASET --ldr_img_dir $LDR_IMG_DIR \
        --ckpt checkpoints/$EXP/last_0.ckpt --crf_basis $CRF_BASIS

# refine emitter 
python train_emitter.py --experiment_name $EXP \
        --max_epochs 1 --dir_val val_0_emitter \
        --ckpt_path checkpoints/$EXP/last_0.ckpt \
        --voxel_path checkpoints/$EXP/bake/vslf_0.npz \
        --emitter_path checkpoints/$EXP/bake/emitter.pth \
        --dataset $DATASET $DATASET_ROOT$SCENE \
        --has_part $HAS_PART --ldr_img_dir $LDR_IMG_DIR  --val_frame $VAL_FRAME\
        --SPP $SPP --spp $spp --crf_basis $CRF_BASIS

mv checkpoints/$EXP/last.ckpt checkpoints/$EXP/last_0.ckpt

# extract emitter
python extract_emitter_ldr.py --mode update\
        --scene $DATASET_ROOT$SCENE\
        --output checkpoints/$EXP/bake\
        --ckpt checkpoints/$EXP/last_0.ckpt\
        --dataset $DATASET --ldr_img_dir $LDR_IMG_DIR

# # refine shading 
python refine_shading.py \
        --scene $DATASET_ROOT$SCENE --dataset $DATASET \
        --ldr_img_dir $LDR_IMG_DIR \
        --slf_path checkpoints/$EXP/bake/vslf_0.npz \
        --emitter_path checkpoints/$EXP/bake/emitter.pth \
        --ckpt checkpoints/$EXP/last_0.ckpt \
        --output outputs/$EXP/shading

# optimize BRDF, CRF
python train_brdf_crf.py --experiment_name $EXP \
        --max_epochs 4 --dir_val val_1 \
        --ckpt_path checkpoints/$EXP/init.ckpt \
        --voxel_path checkpoints/$EXP/bake/vslf_0.npz \
        --emitter_path checkpoints/$EXP/bake/emitter.pth \
        --cache_dir outputs/$EXP/shading \
        --dataset $DATASET $DATASET_ROOT$SCENE \
        --has_part $HAS_PART --ldr_img_dir $LDR_IMG_DIR  --val_frame $VAL_FRAME\
        --SPP $SPP --spp $spp --lp 0.005 --la 0.01 --l_crf_weight 0.001 --crf_basis $CRF_BASIS

mv checkpoints/$EXP/last.ckpt checkpoints/$EXP/last_1.ckpt