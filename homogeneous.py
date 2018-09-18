from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def to_homogeneous(xyz, axis=-1):
    with tf.name_scope('to_homogeneous'):
        xyz = tf.convert_to_tensor(xyz)
        shape = xyz.shape
        ndims = shape.ndims
        n = shape[axis].value
        if n != 3:
            raise ValueError('xyz.shape[axis] must be 3, got %s' % n)
        begin = [0 for _ in range(ndims)]
        size = [-1 for _ in range(ndims)]
        size[axis] = 1
        ones = tf.ones_like(tf.slice(xyz, begin, size))
        return tf.concat((xyz, ones), axis=axis)


def from_homogeneous(xyzw, axis=-1):
    with tf.name_scope('from_homogeneous'):
        xyzw = tf.convert_to_tensor(xyzw)
        n = xyzw.shape[axis].value
        if n != 4:
            raise ValueError('xyzw.shape[axis] must be 4, got %s' % n)
        xyz, w = tf.split(xyzw, [3, 1], axis=-1)
        return xyz / w


def transform(xyzw, T):
    with tf.name_scope('homogeneous_transform'):
        return tf.matmul(xyzw, T, transpose_b=True)


def origin_homogeneous(dtype=tf.float32):
    return tf.constant([0, 0, 0, 1], dtype=dtype, name='homogeneous_origin')


def split_homogeneous(A, squeeze_t=True):
    """Split 4x4 homogeneous transform into rotation R and translation t."""
    with tf.name_scope('split_homogeneous_transform'):
        A = A[..., :3, :]
        R, t = tf.split(A, [3, 1], axis=-1)
        if squeeze_t:
            t = tf.squeeze(t, axis=-1)
    return R, t


def _merge_nonhomogeneous(R, t=None):
    """Create 4x4 homogeneous transform from rotation R and translation t."""
    batch_size = tf.shape(R)[0]
    if t is None:
        t = origin_homogeneous(dtype=R.dtype)
        t = tf.expand_dims(t, axis=0)
        t = tf.tile(t, (batch_size, 1))
    else:
        t = tf.concat(
            (t, tf.ones(shape=(batch_size, 1), dtype=t.dtype)), axis=-1)

    t = tf.expand_dims(t, axis=-1)

    R = tf.concat(
        (R, tf.zeros(shape=(batch_size, 1, 3), dtype=R.dtype)), axis=-2)

    T = tf.concat((R, t), axis=-1)
    return T


def merge_nonhomogeneous(R=None, t=None):
    with tf.name_scope('merge_nonhomogeneous_transform'):
        if R is None:
            if t is None:
                raise ValueError('At least one or (`R`, `t`) must be given')
            return homogeneous_translation(t)
        rs = R.shape
        if rs.ndims < 2 or rs[-1].value != 3 or rs[-2] != 3:
            raise ValueError(
                'R must be shape[-2:] == (3, 3), got %s' % str(rs))
        batch_shape = tf.shape(R)[:-2]
        R = tf.reshape(R, (-1, 3, 3))
        if t is not None:
            ts = t.shape
            if ts.ndims == 0 or ts[-1].value != 3:
                raise ValueError(
                    't must be shape (batch_size=%s, 3), got %s'
                    % (rs[0].value, str(ts)))
            t = tf.reshape(t, (-1, 3))

        T = _merge_nonhomogeneous(R, t)
        T = tf.reshape(T, tf.concat((batch_shape, [4, 4]), axis=0))
        return T


def homogeneous_translation(t):
    with tf.name_scope('homogeneous_translation'):
        t = to_homogeneous(t)
        eye = tf.eye(num_rows=4, num_columns=3, batch_shape=tf.shape(t)[:-1])
        T = tf.concat((eye, tf.expand_dims(axis=-1)), axis=-1)
    return T


def transform_homogeneous(points, T):
    """
    Transform homogeneous `points` according to homogeneous transform T.

    Note to transform a transform, use `compound_transform`

    Args:
        points: (k+1)-D tensor of shape (N_0, ..., N_{k-1}, 4)
        T: (k+1)-D tensor of shape(N_0, ..., N_{k-2}, 4, 4)
    Returns:
        transformed points, same shape as `points`.
    """
    with tf.name_scope('transform_homogeneous'):
        return tf.matmul(points, T, transpose_b=True)


def compound_transform(T0, T1):
    """Get the transform associated with transfrom T0 followed by T1."""
    with tf.name_scope('compound_transform'):
        return tf.matmul(T1, T0)
