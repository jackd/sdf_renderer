# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collection of TF functions for managing 3D camera matrices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf


def perspective(aspect_ratio, fov_y, near_clip, far_clip):
    """
    Computes perspective transformation matrices.

    Functionality mimes gluPerspective (third_party/GL/glu/include/GLU/glu.h).

    Args:
      aspect_ratio: float value specifying the image aspect ratio
        (width/height).
      fov_y: 1-D float32 Tensor with shape [batch_size] specifying output
          vertical field of views in degrees.
      near_clip: 1-D float32 Tensor with shape [batch_size] specifying near
          clipping plane distance.
      far_clip: 1-D float32 Tensor with shape [batch_size] specifying far
          clipping plane distance.

    Returns:
      A [batch_size, 4, 4] float tensor that maps from right-handed points in
      eye space to left-handed points in clip space.
    """
    # The multiplication of fov_y by pi/360.0 simultaneously converts to
    # radians and adds the half-angle factor of .5.
    focal_lengths_y = 1.0 / tf.tan(fov_y * (math.pi / 360.0))
    depth_range = far_clip - near_clip
    p_22 = -(far_clip + near_clip) / depth_range
    p_23 = -2.0 * (far_clip * near_clip / depth_range)

    zeros = tf.zeros_like(p_23, dtype=tf.float32)
    # pyformat: disable
    perspective_transform = tf.concat(
        [
            focal_lengths_y / aspect_ratio, zeros, zeros, zeros,
            zeros, focal_lengths_y, zeros, zeros,
            zeros, zeros, p_22, p_23,
            zeros, zeros, -tf.ones_like(p_23, dtype=tf.float32), zeros
        ], axis=0)
    # pyformat: enable
    perspective_transform = tf.reshape(perspective_transform, [4, 4, -1])
    return tf.transpose(perspective_transform, [2, 0, 1])


def look_at(eye, center, world_up):
    """
    Computes camera viewing matrices.

    Functionality mimes gluLookAt (third_party/GL/glu/include/GLU/glu.h).

    Args:
      eye: 2-D float32 tensor with shape [batch_size, 3] containing the XYZ
          world space position of the camera.
      center: 2-D float32 tensor with shape [batch_size, 3] containing a
          position along the center of the camera's gaze.
      world_up: 2-D float32 tensor with shape [batch_size, 3] specifying the
          world's up direction; the output camera will have no tilt with
          respect to this direction.

    Returns:
      A [batch_size, 4, 4] float tensor containing a right-handed camera
      extrinsics matrix that maps points from world space to points in eye
      space.
    """
    batch_size = center.shape[0].value
    # vector_degeneracy_cutoff = 1e-6
    forward = center - eye
    forward_norm = tf.norm(forward, ord='euclidean', axis=1, keep_dims=True)
    # tf.assert_greater(
    #     forward_norm,
    #     vector_degeneracy_cutoff,
    #     message='Camera matrix is degenerate because '
    #             'eye and center are close.')
    forward = tf.divide(forward, forward_norm)

    to_side = tf.cross(forward, world_up)
    to_side_norm = tf.norm(to_side, ord='euclidean', axis=1, keep_dims=True)
    # tf.assert_greater(
    #     to_side_norm,
    #     vector_degeneracy_cutoff,
    #     message='Camera matrix is degenerate because up and gaze are close '
    #     'or because up is degenerate.')
    to_side = tf.divide(to_side, to_side_norm)
    cam_up = tf.cross(to_side, forward)

    w_column = tf.constant(
        batch_size * [[0., 0., 0., 1.]], dtype=tf.float32)  # [batch_size, 4]
    w_column = tf.reshape(w_column, [batch_size, 4, 1])
    view_rotation = tf.stack(
        [to_side, cam_up, -forward,
         tf.zeros_like(to_side, dtype=tf.float32)],
        axis=1)  # [batch_size, 4, 3] matrix
    view_rotation = tf.concat(
        [view_rotation, w_column], axis=2)  # [batch_size, 4, 4]

    identity_batch = tf.tile(tf.expand_dims(tf.eye(3), 0), [batch_size, 1, 1])
    view_translation = tf.concat([identity_batch, tf.expand_dims(-eye, 2)], 2)
    view_translation = tf.concat(
        [view_translation,
         tf.reshape(w_column, [batch_size, 1, 4])], 1)
    camera_matrices = tf.matmul(view_rotation, view_translation)
    return camera_matrices


def euler_matrices(angles):
    """Computes a XYZ Tait-Bryan (improper Euler angle) rotation.

    Returns 4x4 matrices for convenient multiplication with other
    transformations.

    Args:
      angles: a [batch_size, 3] tensor containing X, Y, and Z angles in
        radians.

    Returns:
      a [batch_size, 4, 4] tensor of matrices.
    """
    s = tf.sin(angles)
    c = tf.cos(angles)
    # Rename variables for readability in the matrix definition below.
    c0, c1, c2 = tf.unstack(c, axis=-1)
    s0, s1, s2 = tf.unstack(s, axis=-1)

    zeros = tf.zeros_like(s0)
    ones = tf.ones_like(s0)

    flattened = tf.concat(
        [
            c2 * c1, c2 * s1 * s0 - c0 * s2, s2 * s0 + c2 * c0 * s1, zeros,
            c1 * s2, c2 * c0 + s2 * s1 * s0, c0 * s2 * s1 - c2 * s0, zeros,
            -s1, c1 * s0, c1 * c0, zeros,
            zeros, zeros, zeros, ones
        ],
        axis=-1)
    reshaped = tf.reshape(flattened, [-1, 4, 4])
    return reshaped


def get_camera_rays(image_height, image_width, fov_y):
    with tf.name_scope('camera_rays'):
        focal_length = image_height / (2*tf.tan(fov_y*math.pi/360))
        h2 = image_height / 2
        w2 = image_width / 2
        hs = tf.linspace(-h2, h2, image_height)
        ws = tf.linspace(-w2, w2, image_width)
        x, y = tf.meshgrid(hs, ws, indexing='ij')
        tile_shape = (tf.shape(fov_y)[0], 1, 1)
        x = tf.tile(tf.expand_dims(x, axis=0), tile_shape)
        y = tf.tile(tf.expand_dims(y, axis=0), tile_shape)
        z = -tf.ones_like(x) * tf.reshape(focal_length, (-1, 1, 1))
        rays = tf.stack((x, y, z), axis=-1)
        rays = rays / tf.norm(rays, axis=-1, keepdims=True)
        return tf.reshape(rays, (-1, image_height*image_width, 3))


def get_transformed_camera_rays(
        image_height, image_width, fov_y, rotation_matrix):
    """Get rays in world coordinates."""
    from .frame import coordinate_transform
    directions = get_camera_rays(image_height, image_width, fov_y)
    directions = coordinate_transform(directions, rotation_matrix)
    return directions


def split_homogeneous(A):
    with tf.name_scope('split_homogeneous'):
        A = A[..., :3, :]
        R, t = tf.split(A, [3, 1], axis=-1)
        t = tf.squeeze(t, axis=-1)
    return R, t
