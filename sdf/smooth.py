"""
Various smoothing algorithms for merging sdfs.

See http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .core import Sdf


def reduce_exponential_min(s, k=32, axis=-1):
    nk = -k
    return tf.log(tf.reduce_sum(tf.exp(nk*s), axis=axis)) / nk


# def reduce_power_min(s, k=8, axis=-1):
#     raise NotImplementedError('bugs...')
#     nk = -k
#     x = tf.reduce_sum(tf.sign(s)*tf.pow(tf.abs(s), nk), axis=axis)
#     return tf.sign(x)*tf.pow(tf.abs(x), 1/nk)


def _min_to_max(min_reduction):
    def f(s, *args, **kwargs):
        return -min_reduction(-s, *args, **kwargs)
    return f


def _reduce_to_ops(reduction):
    def f(*operands, **kwargs):
        return reduction(
            tf.stack(*operands, axis=-1), axis=-1, **kwargs)
    return f


exponential_min = _reduce_to_ops(reduce_exponential_min)
# power_min = _reduce_to_ops(reduce_power_min)


reduce_exponential_max = _min_to_max(reduce_exponential_min)
# reduce_power_max = _min_to_max(reduce_power_min)


class SmoothBox(Sdf):
    def __init__(
            self, dimensions, smoother='exponential', k=32,
            name='smooth_box'):
        self.name = name
        self.smoother = smoother
        with tf.name_scope('%s_name' % name):
            self._dimensions = tf.convert_to_tensor(
                dimensions, dtype=tf.float32, name='dimensions')
            self._k = tf.convert_to_tensor(k, dtype=tf.float32, name='k')

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def k(self):
        return self._k

    def __call__(self, x):
        with tf.name_scope(self.name):
            d = tf.abs(x) - self.dimensions
            k = self.k
            if self.smoother == 'exponential':
                return reduce_exponential_max(d, k, axis=-1)
            # elif self.smoother == 'power':
            #     return reduce_power_max(d, k, axis=-1)
            elif self.smoother is None:
                return tf.reduce_max(d, axis=-1)
            else:
                raise ValueError('Unrecognized smoother "%s"' % self.smoother)
