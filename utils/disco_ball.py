# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import mitsuba

def sample_points_on_sphere(num_points, phase=0):
    """Uniformly sample points on a unit sphere using the Fibonacci lattice method."""
    points = []
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    for i in range(num_points):
        theta = 2 * np.pi * i / phi  # Longitude angle
        z = 1 - (2 * i + 1) / num_points  # Latitude adjustment for uniform spacing
        radius = np.sqrt(1 - z * z)  # Radius of the circle at latitude z

        x = radius * np.cos(theta + phase)
        y = radius * np.sin(theta + phase)
        points.append([x, y, z])

    return np.array(points)

def make_disco_ball(
        scene_dict: dict,
        position, 
        radius,
        light_intensity,
        light_num=20,
        light_radius_rate=0.1,
        spot_intensity=10,
        spot_cutoff_angle=20.0,
        phase=0):
    
    colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    position = np.array(position)
    points_unit = sample_points_on_sphere(light_num, phase)
    light_radius = radius * light_radius_rate
    distance_from_center = radius - light_radius * 0.6
    points_sphere = points_unit * distance_from_center + position

    # make center ball
    scene_dict.update({
        'disco_ball': {
            'type': 'sphere',
            'to_world': mitsuba.ScalarTransform4f\
                      .translate(position.tolist())\
                      .scale([radius, radius, radius]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.2, 0.2, 0.2]
                }
            },
        }
    })

    # make lights
    for i in range(light_num):
        light_pos = points_sphere[i]
        light_value = colors[i % colors.shape[0]] * light_intensity

        light_config = {
            'type': 'sphere',
            'to_world': mitsuba.ScalarTransform4f\
                      .translate(light_pos.tolist())\
                      .scale([light_radius, light_radius, light_radius]),
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': light_value.tolist()
                }
            }
        }

        spot_o = points_unit[i] * (radius + light_radius) + position
        spot_t = spot_o + points_unit[i]
        spot_value = colors[i % colors.shape[0]] * spot_intensity
        spot_config = {
            'type': 'spot',
            'to_world': mitsuba.ScalarTransform4f.look_at(
                    origin=spot_o.tolist(),
                    target=spot_t.tolist(),
                    up=[0, 0, 1]
                ),    # Light direction
            'cutoff_angle': spot_cutoff_angle,        # Defines the angle of the light cone         # Inner angle with full intensity
            'intensity': {
                'type': 'rgb',
                'value': spot_value.tolist()
            }
        }


        scene_dict.update({
            'light_{}'.format(i): light_config,
            'spot_{}'.format(i): spot_config
        })