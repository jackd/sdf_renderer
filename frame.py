from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def coordinate_transform(x, R=None, t=None):
    """
    Perform inverse transformation of points x.

    Args:
        x: (k+2)-D tensor shape [N_0, ..., N_{k-1}, P, 3]
        R: (k+2)-D rotation tensor shape [N_0, ..., N_{k-1}, 3, 3]
        t: (k+1)-D translation tensor shape [N_0, ..., N_{k-1}, 3]

    Returns:
        (k+2)-D tensor, same shape as x.
    """
    with tf.name_scope('coordinate_transform'):
        # if R is not None:
        #     if len(x.shape) == 2:
        #         x = tf.einsum('ijk,ik->ij', R, x)
        #     else:
        #         x = tf.matmul(x, R)
        if R is not None:
            x = tf.matmul(x, R, transpose_b=True)
        if t is not None:
            x = x + t
        return x


def inverse_coordinate_transform(x, R=None, t=None):
    """
    Perform inverse transformation of points x.

    Args:
        x: (k+2)-D tensor shape [N_0, ..., N_{k-1}, P, 3]
        R: (k+2)-D rotation tensor shape [N_0, ..., N_{k-1}, 3, 3]
        t: (k+1)-D translation tensor shape [N_0, ..., N_{k-1}, 3]
    Returns:
        (k+2)-D tensor, same shape as x.
    """
    with tf.name_scope('inverse_coordinate_transform'):
        if t is not None:
            x = x - t
        # if R is not None:
        #     if len(x.shape) == 2:
        #         x = tf.einsum('ijk,ij->ik', R, x)
        #     else:
        #         x = tf.matmul(x, R, transpose_b=True)
        if R is not None:
            x = tf.matmul(x, R, transpose_b=False)
    return x


def get_inverse_transform(R, t):
    """
    Get the inverse 4x4 homogeneous transform of T.

    Args:
        R: (k+2)-D tensor of shape (N_0, ..., N_{k-1}, 3, 3)
        t: (k+1)-D tensor of shape (N_0, ..., N_{k-1}, 3)

    Returns:
        Transformed versions of `R, t`, same shape
    """
    with tf.name_scope('inverse_homogeneous_transform'):
        if R.shape[-2:].as_list() != [3, 3]:
            raise ValueError(
                'Invalid shape for rotation matrix: %s' % str(R.shape))
        if t.shape[-1].value != 3:
            raise ValueError(
                'Invalid shape for translation: %s' % str(t.shape))
        t2 = -tf.reduce_sum(R*t, axis=-2)
        n_dims = R.shape.ndims
        batch_shape = tuple(range(n_dims-2))
        R2 = tf.transpose(R, batch_shape + (n_dims-1, n_dims-2))
    return R2, t2
