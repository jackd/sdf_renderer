from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def coordinate_transform(x, R=None, t=None):
    with tf.name_scope('coordinate_transform'):
        if R is not None:
            if len(x.shape) == 2:
                x = tf.einsum('ijk,ik->ij', R, x)
            else:
                x = tf.matmul(x, R)
        if t is not None:
            x = x + t
        return x


def inverse_coordinate_transform(x, R=None, t=None):
    with tf.name_scope('inverse_coordinate_transform'):
        if t is not None:
            x = x - t
        if R is not None:
            if len(x.shape) == 2:
                x = tf.einsum('ijk,ij->ik', R, x)
            else:
                x = tf.matmul(x, R, transpose_b=True)
    return x
