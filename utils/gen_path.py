# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from .ray_utils import generate_interpolated_path
from .dataset.synthetic_ldr import SyntheticDatasetLDR
from .dataset.real_ldr import RealDatasetLDR
from .dataset.scannetpp.dataset import Scannetpp

def generate_path_kitchen():
    root_dir = '/hdd/datasets/fipt/indoor_synthetic/kitchen'
    dataset = SyntheticDatasetLDR(root_dir, split='val', pixel=True)
    poses = np.array(dataset.poses)
    # print('poses:', poses.shape) #(n, 3, 4)
    
    key_idx = [11, 12]
    pose_keyframe = poses[key_idx]
    render_traj = generate_interpolated_path(pose_keyframe, n_interp=300)
    output_path = os.path.join(root_dir, 'render_traj.npy')
    np.save(output_path, render_traj)
    print('Save', output_path)

def generate_path_bathroom():
    root_dir = '/hdd/datasets/fipt/indoor_synthetic/bathroom'
    dataset = SyntheticDatasetLDR(root_dir, split='val', pixel=True)
    poses = np.array(dataset.poses)

    key_idx = [10, 12]
    pose_keyframe = poses[key_idx]
    render_traj = generate_interpolated_path(pose_keyframe, n_interp=300)
    output_path = os.path.join(root_dir, 'render_traj.npy')
    np.save(output_path, render_traj)
    print('Save', output_path)

def generate_path_bedroom():
    root_dir = '/hdd/datasets/fipt/indoor_synthetic/bedroom'
    dataset = SyntheticDatasetLDR(root_dir, split='val', pixel=True)
    poses = np.array(dataset.poses)

    key_idx = [12, 10]
    pose_keyframe = poses[key_idx]
    render_traj = generate_interpolated_path(pose_keyframe, n_interp=300)
    output_path = os.path.join(root_dir, 'render_traj.npy')
    np.save(output_path, render_traj)
    print('Save', output_path)

def generate_path_livingroom():
    root_dir = '/hdd/datasets/fipt/indoor_synthetic/livingroom'
    dataset = SyntheticDatasetLDR(root_dir, split='val', pixel=True)
    poses = np.array(dataset.poses)

    key_idx = [10, 13]
    pose_keyframe = poses[key_idx]
    render_traj = generate_interpolated_path(pose_keyframe, n_interp=300)
    output_path = os.path.join(root_dir, 'render_traj.npy')
    np.save(output_path, render_traj)
    print('Save', output_path)

def generate_path_classroom():
    root_dir = '/hdd/datasets/fipt/real/classroom'
    dataset = RealDatasetLDR(root_dir, split='train', pixel=True)
    poses = np.array(dataset.C2Ws)

    key_idx = [76, 37]
    pose_keyframe = poses[key_idx]
    render_traj = generate_interpolated_path(pose_keyframe, n_interp=300)
    output_path = os.path.join(root_dir, 'render_traj.npy')
    np.save(output_path, render_traj)
    print('Save', output_path)

def generate_path_conferenceroom():
    root_dir = '/hdd/datasets/fipt/real/conferenceroom'
    dataset = RealDatasetLDR(root_dir, split='train', pixel=True)
    poses = np.array(dataset.C2Ws)

    key_idx = [30, 42, 108]
    pose_keyframe = poses[key_idx]
    render_traj = generate_interpolated_path(pose_keyframe, n_interp=150)
    output_path = os.path.join(root_dir, 'render_traj.npy')
    np.save(output_path, render_traj)
    print('Save', output_path)

def generate_path_scannetpp_bathroom2():
    root_dir = '/hdd/datasets/scannetpp'
    scene_id = '45b0dac5e3'
    split = 'all'
    n_frames = 150
    key_idx = [6, 227, 147]
    dataset = Scannetpp(root_dir, scene_id, split=split)
    poses = np.array(dataset.C2Ws)

    pose_keyframe = poses[key_idx]
    render_traj = generate_interpolated_path(pose_keyframe, n_interp=n_frames)
    output_path = os.path.join(root_dir, 'data', scene_id, 'psdf', 'render_traj.npy')
    np.save(output_path, render_traj)
    print('Save', output_path)

def generate_path_scannetpp_room2():
    root_dir = '/hdd/datasets/scannetpp'
    scene_id = '7e09430da7'
    split = 'train'
    n_frames = 225
    key_idx = [200, 121, 96]
    dataset = Scannetpp(root_dir, scene_id, split=split)
    poses = np.array(dataset.C2Ws)

    pose_keyframe = poses[key_idx]
    render_traj = generate_interpolated_path(pose_keyframe, n_interp=n_frames)[-300:]
    output_path = os.path.join(root_dir, 'data', scene_id, 'psdf', 'render_traj.npy')
    np.save(output_path, render_traj)
    print('Save', output_path)

if __name__ == '__main__':
    generate_path_scannetpp_room2()