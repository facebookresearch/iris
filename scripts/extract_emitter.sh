# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



python -m utils.extract_emitter_mesh \
    --mesh_scene   /hdd/datasets/scannetpp/data/7e09430da7/scans/scene.ply \
    --emitter      /hdd/meta/meta_iris_dev/outputs/240506_scannetpp_room2/bake/emitter.pth \
    --mesh_emitter /hdd/meta/meta_iris_dev/outputs/240506_scannetpp_room2/bake/emitter.ply 

# Emitter average radiance:
# bathroom2: [ 9.040693,  9.697464, 10.583247]
# conferenceroom: [2.1285791, 2.1815128, 2.379316 ]
# room2: [8.679976, 8.248041, 2.860989]