#!/usr/bin/python
"""Script checking coordinate transforms."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sdf_renderer.camera_utils as camera_utils
import sdf_renderer.homogeneous as homogeneous
import sdf_renderer.frame as frame

tf.enable_eager_execution()

eye = tf.constant([[1, 1, 1]], dtype=tf.float32)
center = tf.zeros(shape=(1, 3), dtype=tf.float32)
world_up = tf.constant([[0, 0, 1]], dtype=tf.float32)

camera_matrices = camera_utils.look_at(eye, center, world_up)

R, t = homogeneous.split_homogeneous(camera_matrices)

eye_in_camera_coords = tf.squeeze(frame.coordinate_transform(
      tf.expand_dims(eye, axis=1), R, t), axis=1)

coords = tf.random_normal(shape=(1, 10, 3), dtype=tf.float32)
transformed = frame.coordinate_transform(coords, R, t)
inv = frame.inverse_coordinate_transform(transformed, R, t)
transform_err = inv - coords

coords0 = coords[:, 0:1]
transformed0 = frame.coordinate_transform(coords0, R, t)
inv0 = frame.inverse_coordinate_transform(transformed0, R, t)
transform_err0 = inv0 - coords0

# with tf.Session() as sess:
#     i, te, te0 = sess.run(
#         (eye_in_camera_coords, transform_err, transform_err0))

print('offset in camera coordinates (should be zeros): %s'
      % str(eye_in_camera_coords))
print('transform_error (rank 3): %s' % str(np.max(np.abs(transform_err))))
# print('transform_error (rank 2): %s' % str(np.max(np.abs(transform_err0))))
