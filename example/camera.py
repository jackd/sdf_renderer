#!/usr/bin/python
"""Script for visualizing the camera rays."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sdf_renderer.camera_utils as camera_utils
import sdf_renderer.homogeneous as homogeneous

tf.enable_eager_execution()

eye = tf.constant([1, 1, 1], dtype=tf.float32)
center = tf.zeros(shape=(3,), dtype=tf.float32)
world_up = tf.constant([0, 0, 1], dtype=tf.float32)

aspect_ratio = tf.ones(shape=(), dtype=tf.float32)
# fov_y = tf.ones(shape=(), dtype=tf.float32) * 40.0
sensor_width = tf.ones(shape=(), dtype=tf.float32) * 32.0
focal_length = tf.ones(shape=(), dtype=tf.float32) * 35.0
near_clip = tf.ones(shape=(), dtype=tf.float32) * 0.01
far_clip = tf.ones(shape=(), dtype=tf.float32)


image_height = 8
image_width = 8
focal_length_px = camera_utils.get_focal_length_px(
    image_width, sensor_width, focal_length)

normalization = 'z'


def vis(offset, direction):
    from sdf_renderer.vis import vis_rays, vis_axes, vis_points, show

    offset = offset
    direction = direction
    vis_points(offset + direction, color=(1, 0, 0), scale_factor=0.05)
    vis_points(offset, scale_factor=0.1, color=(0, 0, 1))

    vis_axes()
    print(direction.shape)
    print(direction[:2, :2])
    vis_rays(offset, direction)

    show()


camera_matrices = camera_utils.look_at(eye, center, world_up)

R, t = homogeneous.split_homogeneous(camera_matrices)

directions = camera_utils.get_transformed_camera_rays(
    image_height, image_width, focal_length_px, R, normalization=normalization)


# with tf.Session() as sess:
#     i, d = sess.run((eye, directions))
#
vis(np.array(eye), np.array(directions))
