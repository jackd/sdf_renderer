#!/usr/bin/python
"""Script for checking numerics under certain parameter settings."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
# from sdf_renderer.sdf.primitives import Box
# from sdf_renderer.sdf.smooth import SmoothBox
from sdf_renderer.sdf.primitives import Sphere
from sdf_renderer import camera_utils
from sdf_renderer import render

threshold = 1e-7
max_length = 3.0
# image_height = 10
# image_width = 10
image_height = 27
image_width = 27
# image_height = 28
# image_width = 28
# image_height = 29
# image_width = 29

# image_height = 34
# image_width = 34


def build_graph():
    sphere = Sphere(radius=0.3).translate((0.2, 0, 0))
    # box = Box((0.1, 0.2, 0.3)).translate((0, -0.1, 0.1))
    # box = SmoothBox((0.1, 0.2, 0.3)).translate((0, -0.1, 0.1))
    # sdf = sphere.union(box)
    sdf = sphere
    # sdf = box
    # translation = -tf.ones(shape=(3,), dtype=tf.float32)
    # sdf = sdf.translate(translation)
    scale_factor = tf.constant(0.8, dtype=tf.float32)
    sdf = sdf.scale(scale_factor)

    eye = tf.constant([[1, 1, 1]], dtype=tf.float32)
    center = tf.zeros(shape=(1, 3), dtype=tf.float32)
    world_up = tf.constant([[0, 0, 1]], dtype=tf.float32)

    fov_y = tf.ones(shape=(1,), dtype=tf.float32) * 40.0

    camera_matrices = camera_utils.look_at(eye, center, world_up)

    R, t = camera_utils.split_homogeneous(camera_matrices)

    directions = camera_utils.get_transformed_camera_rays(
        image_height, image_width, fov_y, R)

    return sdf, eye, directions, scale_factor


graph = tf.Graph()
with graph.as_default():
    sdf, eye, directions, scale_factor = build_graph()
    kwargs = dict(threshold=threshold)
    lengths, points, hit, missed = render.get_intersections(
        sdf, eye, directions, max_length, back_prop=True,
        fix_lengths=False, threshold=threshold)
    # lengths = tf.identity(lengths)
    grad, = tf.gradients(tf.boolean_mask(lengths, hit), scale_factor)

with tf.Session(graph=graph) as sess:
    # l0, g = sess.run((lengths, grad))
    # print(l0, g)
    l0, g0, h0 = sess.run((lengths, grad, hit))


graph = tf.Graph()
with graph.as_default():
    sdf, eye, directions, scale_factor = build_graph()
    # lengths, passed = render.get_linearized_solution_lengths(
    #     eye, directions, l0, sdf)
    lengths = render.fix_length_gradient(
        eye, directions, l0, sdf)
    # lengths, passed = render.fix_length_gradient(
    #     eye, directions, l0, sdf)
    # passed = tf.Print(
    #     passed,
    #     [tf.reduce_sum(tf.cast(tf.logical_and(h0, passed), tf.uint8))])
    # hit = tf.logical_and(h0, tf.logical_not(passed))
    lengths = tf.minimum(max_length, lengths)
    grad, = tf.gradients(tf.boolean_mask(lengths, h0), scale_factor)
    with tf.control_dependencies([tf.add_check_numerics_ops()]):
        grad = tf.identity(grad)
#
with tf.Session(graph=graph) as sess:
    l1, g = sess.run((lengths, grad))
    print(np.max(np.abs(l0 - l1)), g, np.abs((g-g0)/g0))
