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


def _assert_same_shape(x, y, key_x, key_y):
    if x.shape != y.shape:
        raise ValueError(
            '`%s` and `%s` must have same shape, got %s and %s'
            % (key_x, key_y, str(x), str(y.shape)))


def _perspective(
        aspect_ratio, fov_y, near_clip, far_clip, dtype=tf.float32,
        angle_units='rad'):
    """Return shape is `(batch_size, 4*4)`."""
    # The multiplication of fov_y by pi/360.0 simultaneously converts to
    # radians and adds the half-angle factor of .5.
    # focal_lengths_y = 1.0 / tf.tan(fov_y * (math.pi / 360.0))
    focal_lengths_y = get_focal_length(1.0, fov_y, angle_units=angle_units)
    depth_range = far_clip - near_clip
    p_22 = -(far_clip + near_clip) / depth_range
    p_23 = -2.0 * (far_clip * near_clip / depth_range)

    zeros = tf.zeros_like(p_23, dtype=dtype)
    # pyformat: disable
    transform_values = [
        focal_lengths_y / aspect_ratio, zeros, zeros, zeros,
        zeros, focal_lengths_y, zeros, zeros,
        zeros, zeros, p_22, p_23,
        zeros, zeros, -tf.ones_like(p_23, dtype=dtype), zeros
    ]
    # perspective_transform = tf.concat(transform_values, axis=0)
    # # pyformat: enable
    # perspective_transform = tf.reshape(perspective_transform, [4, 4, -1])
    # return tf.transpose(perspective_transform, [2, 0, 1])
    perspective_transform = tf.stack(transform_values, axis=-1)
    # perspective_transform = tf.reshape(
    #     perspective_transform, (-1, 4, 4))
    return perspective_transform


def perspective(
        aspect_ratio, fov_y, near_clip, far_clip, dtype=tf.float32,
        angle_units='rad'):
    """
    Computes perspective transformation matrices.

    Functionality mimes gluPerspective (third_party/GL/glu/include/GLU/glu.h).

    Args:
      aspect_ratio: float value specifying the image aspect ratio
        (width/height).
      fov_y: k-D float32 Tensor with shape [N_0, ..., N_{k-1}] specifying
          output vertical field of views in degrees.
      near_clip: k-D float32 Tensor with shape [N_0, ..., N_{k-1}] specifying
          near clipping plane distance.
      far_clip: K-D float32 Tensor with shape [N_0, ..., N_{k-1}] specifying
          clipping plane distance.
      dtype: data type of returned array

    Returns:
      A [N_0, ..., N_{k-1}, 4, 4] tensor that maps from right-handed
      points in eye space to left-handed points in clip space.
    """
    with tf.name_scope('perspective_transform'):
        fov_y = tf.convert_to_tensor(fov_y, dtype=dtype)
        near_clip = tf.convert_to_tensor(near_clip, dtype=dtype)
        far_clip = tf.convert_to_tensor(far_clip, dtype=dtype)
        _assert_same_shape(fov_y, near_clip, 'fov_y', 'near_clip')
        _assert_same_shape(fov_y, far_clip, 'fov_y', 'far_clip')
        s = tf.concat((tf.shape(fov_y), (4, 4)), 0)
        fov_y = tf.reshape(fov_y, (-1,))
        near_clip = tf.reshape(near_clip, (-1,))
        far_clip = tf.reshape(far_clip, (-1,))
        transform = _perspective(
            aspect_ratio, fov_y, near_clip, far_clip,
            dtype=dtype, angle_units=angle_units)
        transform = tf.reshape(transform, s)
    return transform


def _look_at(eye, center, world_up):
    """
    Computes camera viewing matrices.

    Functionality mimes gluLookAt (third_party/GL/glu/include/GLU/glu.h).

    Args:
      eye: 2-D float32 tensor with shape [batch_size, 3] containing
          the world space position of the camera.
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
    from .homogeneous import origin_homogeneous
    batch_size = tf.shape(eye)[0]
    # vector_degeneracy_cutoff = 1e-6
    forward = center - eye
    forward_norm = tf.norm(forward, ord='euclidean', axis=1, keepdims=True)
    # tf.assert_greater(
    #     forward_norm,
    #     vector_degeneracy_cutoff,
    #     message='Camera matrix is degenerate because '
    #             'eye and center are close.')
    forward = tf.divide(forward, forward_norm)

    to_side = tf.cross(forward, world_up)
    to_side_norm = tf.norm(to_side, ord='euclidean', axis=1, keepdims=True)
    # tf.assert_greater(
    #     to_side_norm,
    #     vector_degeneracy_cutoff,
    #     message='Camera matrix is degenerate because up and gaze are close '
    #     'or because up is degenerate.')
    to_side = tf.divide(to_side, to_side_norm)
    cam_up = tf.cross(to_side, forward)

    w_column = tf.tile(
        tf.reshape(origin_homogeneous(), (1, 4, 1)), (batch_size, 1, 1))
    # [batch_size, 4, 1]
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


def look_at(eye, center, world_up):
    """
    Computes camera viewing matrices.

    Functionality mimes gluLookAt (third_party/GL/glu/include/GLU/glu.h).

    Args:
      eye: k-D float32 tensor with shape [N_0, ..., N_{k-2}, 3] containing
          the world space position of the camera.
      center: k-D float32 tensor with shape [N_0, ..., N_{k-2}, 3] containing a
          position along the center of the camera's gaze.
      world_up: k-D float32 tensor with shape [N_0, ..., N_{k-2}, 3] specifying
           the world's up direction; the output camera will have no tilt with
          respect to this direction.

    Returns:
      A [N_0, ..., N_{k-2}, 4, 4] float tensor containing a right-handed
      camera extrinsics matrix that maps points from world space to points in
      eye space.
    """
    with tf.name_scope('look_at'):
        s = tf.shape(eye)[:-1]
        _assert_same_shape(eye, center, 'eye', 'center')
        _assert_same_shape(eye, world_up, 'eye', 'world_up')
        eye = tf.reshape(eye, (-1, 3))
        center = tf.reshape(center, (-1, 3))
        world_up = tf.reshape(world_up, (-1, 3))
        camera_matrices = _look_at(eye, center, world_up)
        final_shape = tf.concat([s, [4, 4]], axis=0)
        camera_matrices = tf.reshape(camera_matrices, final_shape)
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
    with tf.name_scope('euler_matrices'):
        s = tf.sin(angles)
        c = tf.cos(angles)
        # Rename variables for readability in the matrix definition below.
        c0, c1, c2 = tf.unstack(c, axis=-1)
        s0, s1, s2 = tf.unstack(s, axis=-1)

        zeros = tf.zeros_like(s0)
        ones = tf.ones_like(s0)

        flattened = tf.stack(
            [
                c2 * c1, c2 * s1 * s0 - c0 * s2, s2 * s0 + c2 * c0 * s1, zeros,
                c1 * s2, c2 * c0 + s2 * s1 * s0, c0 * s2 * s1 - c2 * s0, zeros,
                -s1, c1 * s0, c1 * c0, zeros,
                zeros, zeros, zeros, ones
            ],
            axis=-1)
        shape = tf.concat([tf.shape(c0), [4, 4]], axis=-1)
        reshaped = tf.reshape(flattened, shape)
    return reshaped


def get_focal_length(dim_length, fov, angle_units='rad'):
    with tf.name_scope('focal_length'):
        if angle_units == 'deg':
            angle = fov * math.pi / 360
        elif angle_units == 'rad':
            angle = fov / 2
        else:
            raise KeyError('Invalid angle_units')
        focal_length = dim_length / (2*tf.tan(angle))
    return focal_length


def get_field_of_view(focal_length, sensor_length, dtype=tf.float32):
    """
    Get the field of view based on focal length and sensor length.

    Inputs must be same size.

    Args:
        focal_length: focal length of camera.
        sensor_length: length of sensor in the appropriate dimension.

    Returns:
        field of view in radians.
    """
    with tf.name_scope('field_of_view'):
        focal_length = tf.convert_to_tensor(focal_length, dtype=dtype)
        sensor_length = tf.convert_to_tensor(sensor_length, dtype=dtype)
        if sensor_length.shape.ndims == focal_length.shape.ndims + 1:
            focal_length = tf.expand_dims(focal_length, axis=-1)
        fov = 2 * tf.atan(sensor_length / (2*focal_length))
    return fov


def get_focal_length_px(dim_length, sensor_length, focal_length):
    """
    Get the focal length in pixels.

    Args:
        dim_length: length of dimension, e.g. image height/width, in pixels.
        sensor_length: length of sensor in the relevant direction in world
            units (e.g. mm).
        focal_length: in world units (e.g. mm).
    Returns:
        focal length in pixels.
    """
    with tf.name_scope('focal_length_in_px'):
        return focal_length * dim_length / sensor_length


def get_camera_rays(
        image_height, image_width, focal_length_px,
        normalization='norm', include_corners=True,
        as_rect=True):
    """
    Get rays for the given camera.

    Args:
        image_height: int, number of pixels vertically
        image_width: int, number of pixels horizontally
        focal_length_px: float tensor, shape `focal_shape` in pixels.
        normalization: string key denoting method of normalization. One of
            `('norm', 'z', 'none')` for division by norm, z-value or nothing.
        include_corners: bool indicating whether the rays should go to corners
            or centres of grid squares.
        as_rect: infuences output shape.

    Returns:
        Float tensor of camera rays, shape
            `focal_shape` + [image_height, image_width, 3]
        if as_rect is True, else
            `focal_shape` + [image_height * image_width, 3]
    """
    with tf.name_scope('camera_rays'):
        if include_corners:
            h2 = image_height / 2
            w2 = image_width / 2
            hs = tf.linspace(-h2, h2, image_height)
            ws = tf.linspace(-w2, w2, image_width)
        else:
            hs = (tf.range(image_height, tf.float32) + 0.5) - h2
            ws = (tf.range(image_width, tf.float32) + 0.5) - w2
        x, y = tf.meshgrid(hs, ws, indexing='ij')
        k = focal_length_px.shape.ndims
        if k > 0:
            focal_shape = tf.shape(focal_length_px)
            tile_shape = tf.concat([focal_shape, [1, 1]], axis=0)
            base_shape = [1]*k + [image_height, image_width]
            x = tf.tile(tf.reshape(x, base_shape), tile_shape)
            y = tf.tile(tf.reshape(y, base_shape), tile_shape)
        else:
            focal_shape = None
        z = -focal_length_px*tf.ones_like(x)
        rays = tf.stack((x, y, z), axis=-1)
        normalization = normalization.lower()
        if normalization == 'norm':
            rays = rays / tf.norm(rays, axis=-1, keepdims=True)
        elif normalization == 'z':
            rays = rays / rays[..., -1:]
        elif normalization == 'none':
            pass
        else:
            raise KeyError('Invalid normalization "%s"' % normalization)
        if not as_rect:
            out_shape = [image_height*image_width, 3]
            if focal_shape is not None:
                out_shape = tf.concat([focal_shape, out_shape], 0)
            rays = tf.reshape(rays, out_shape)
    return rays


def get_transformed_camera_rays(
        image_height, image_width, focal_length_px, rotation_matrix,
        normalization='norm', include_corners=True, as_rect=False):
    """Get rays in world coordinates."""
    # from .frame import coordinate_transform
    from .frame import inverse_coordinate_transform
    with tf.name_scope('transformed_camera_rays'):
        focal_length_px = tf.convert_to_tensor(focal_length_px)
        directions = get_camera_rays(
            image_height, image_width, focal_length_px,
            normalization=normalization, include_corners=include_corners,
            as_rect=False)
        # directions = coordinate_transform(directions, rotation_matrix)
        directions = inverse_coordinate_transform(directions, rotation_matrix)

        if as_rect:
            focal_shape = tf.shape(focal_length_px)
            out_shape = tf.concat(
                (focal_shape, [image_height, image_width, 3]), axis=0)
            directions = tf.reshape(focal_shape, out_shape)
    return directions
