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
SPP=256
spp=16

# render
python render.py --experiment_name $EXP --device 0\
        --ckpt last_1.ckpt \
        --dataset $DATASET $DATASET_ROOT --scene $SCENE\
        --res_scale $RES_SCALE\
        --emitter_path checkpoints/$EXP/bake\
        --output_path 'outputs/'$EXP'/output'\
        --split 'test'\
        --SPP $SPP --spp $spp --crf_basis $CRF_BASIS 

python render_video.py --experiment_name $EXP --device 0\
        --ckpt last_1.ckpt \
        --dataset $DATASET $DATASET_ROOT --scene $SCENE \
        --res_scale $RES_SCALE\
        --emitter_path checkpoints/$EXP/bake\
        --output_path 'outputs/'$EXP'/video'\
        --split 'test'\
        --SPP $SPP --spp $spp --crf_basis $CRF_BASIS  

# relighting 
python render_relight.py --experiment_name $EXP --device 0\
        --ckpt last_1.ckpt --mode traj\
        --dataset $DATASET $DATASET_ROOT --scene $SCENE \
        --res_scale $RES_SCALE \
        --emitter_path checkpoints/$EXP/bake\
        --output_path 'outputs/'$EXP'/relight/video_relight_0'\
        --split 'test'\
        --light_cfg 'configs/scannetpp/room2/relight_0.yaml' \
        --SPP $SPP --spp $spp --crf_basis $CRF_BASIS