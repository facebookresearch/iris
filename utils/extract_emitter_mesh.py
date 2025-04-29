# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import trimesh
import torch
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--mesh_scene', type=str, required=True)
    parser.add_argument('--emitter', type=str, required=True)
    parser.add_argument('--mesh_emitter', type=str, required=True)
    args = parser.parse_args()

    mesh  = trimesh.load(args.mesh_scene)
    vertices = np.array(mesh.vertices).astype(np.float32)
    faces = np.array(mesh.faces).astype(np.int32)

    emitter = torch.load(args.emitter)
    is_emitter = emitter['is_emitter'].numpy()
    emitter_vertices = emitter['emitter_vertices'].numpy()
    emitter_area = emitter['emitter_area'].numpy()
    emitter_radiance = emitter['emitter_radiance'].numpy()

    # calculate average radiance weighted by area
    emitter_radiance = emitter_radiance[:np.sum(is_emitter)]
    emission_avg = np.sum(emitter_radiance * emitter_area[..., None], axis=0) / np.sum(emitter_area)
    print('Average radiance:', emission_avg)

    # Export emitter mesh
    ef = faces[is_emitter]
    ev = vertices[ef]
    n_ef = ef.shape[0]
    ef = ef.flatten()
    u, i = np.unique(ef, return_inverse=True)
    # print(np.abs(u[i] - ef).max())

    v_new = vertices[u]
    f_new = i.reshape(n_ef, 3).astype(np.int32)
    emitter_mesh = trimesh.Trimesh(vertices=v_new, faces=f_new)
    emitter_mesh.export(args.mesh_emitter)
    print('Exported emitter mesh to', args.mesh_emitter)


if __name__ == '__main__':
    main()