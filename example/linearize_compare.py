#!/usr/bin/python
"""Compares values and gradients of linearized intersection lengths."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sdf_renderer.sdf.primitives import Sphere
from sdf_renderer.sdf.primitives import Box
from sdf_renderer import camera_utils
import sdf_renderer.homogeneous as homogeneous
from sdf_renderer import render

threshold = 1e-5

sphere = Sphere(radius=0.3).translate((0.2, 0, 0))
box = Box((0.1, 0.2, 0.3)).translate((0, -0.1, 0.1))
sdf = sphere.union(box)
scale_factor = tf.constant(0.8, dtype=tf.float32)
sdf = sdf.scale(scale_factor)

eye = tf.constant([[1, 1, 1]], dtype=tf.float32)
center = tf.zeros(shape=(1, 3), dtype=tf.float32)
world_up = tf.constant([[0, 0, 1]], dtype=tf.float32)

fov_y = tf.ones(shape=(1,), dtype=tf.float32) * 40.0

image_height = 64
image_width = 64

camera_matrices = camera_utils.look_at(eye, center, world_up)

R, t = homogeneous.split_homogeneous(camera_matrices)

directions = camera_utils.get_transformed_camera_rays(
    image_height, image_width, fov_y, R)


max_length = 3
args = sdf, eye, directions, max_length
kwargs = dict(threshold=threshold)
lengths, points, hit, missed = render.get_intersections(
    *args, back_prop=True, linearize=False, **kwargs)
fixed_lengths, fp, f_hit, f_missed = render.get_intersections(
    *args, back_prop=True, linearize=True, **kwargs)

lengths = tf.boolean_mask(lengths, hit)
fixed_lengths = tf.boolean_mask(fixed_lengths, hit)

g0, = tf.gradients(lengths, scale_factor)
g1, = tf.gradients(fixed_lengths, scale_factor)

gi0, = tf.gradients(lengths, eye)
gi1, = tf.gradients(fixed_lengths, eye)

gf0, = tf.gradients(lengths, fov_y)
gf1, = tf.gradients(fixed_lengths, fov_y)


with tf.Session() as sess:
    gs, gis, gfs, ls = sess.run((
        (g0, g1),
        (gi0, gi1),
        (gf0[0], gf1[0]),
        (lengths, fixed_lengths)))

print('maximum length diff: %s' % str(np.max(np.abs(ls[1] - ls[0]))))
print('Gradients w.r.t scale factor: %s' % str(gs))
print('Gradient diff w.r.t camera offset: %s' % str(gis[0] - gis[1]))
print('Gradients w.r.t fov: %s' % str(gfs))
