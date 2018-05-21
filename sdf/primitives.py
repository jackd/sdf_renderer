"""Provides `Sdf`s for primitive object shapes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .core import Sdf


class Sphere(Sdf):
    def __init__(self, center=(0, 0, 0), radius=1, name='sphere'):
        self.name = name
        with tf.name_scope('%s_setup' % name):
            self._center = tf.convert_to_tensor(
                center, dtype=tf.float32, name='center')
            self._radius = tf.convert_to_tensor(
                radius, dtype=tf.float32, name='radius')

    @property
    def center(self):
        return self._center

    @property
    def radius(self):
        return self._radius

    def __call__(self, x):
        with tf.name_scope(self.name):
            norm = tf.norm(x - self.center, axis=-1)
            return norm - self.radius


class Box(Sdf):
    def __init__(self, dimensions, name='box'):
        self.name = name
        with tf.name_scope('%s_setup' % name):
            self._dimensions = tf.convert_to_tensor(
                dimensions, dtype=tf.float32, name='dimensions')

    @property
    def dimensions(self):
        return self._dimensions

    def __call__(self, x):
        with tf.name_scope(self.name):
            d = tf.abs(x) - self.dimensions
            # Note: the norm version is the actual SDF
            # but has gradient issues?
            return tf.reduce_max(d, axis=-1)
            # return tf.minimum(tf.reduce_max(d, axis=-1), 0) + \
            #     tf.norm(tf.nn.relu(d), axis=-1)


class SdfFn(Sdf):
    def __init__(self, fn):
        self._fn = fn

    def __call__(self, x):
        return self._fn(x)


def convex_hull(W, b, reduction=tf.reduce_max, **kwargs):
    with tf.name_scope('convex_hull'):
        W = tf.convert_to_tensor(W, dtype=tf.float32, name='normals')
        b = tf.convert_to_tensor(b, dtype=tf.float32, name='b')

    def f(x):
        return reduction(tf.matmul(x, W) + b, **kwargs)
    return SdfFn(f)


def convex_hull_union(W, b, n_hulls):
    with tf.name_scope('conver_hull_union'):
        W = tf.convert_to_tensor(W, dtype=tf.float32, name='normals')
        b = tf.convert_to_tensor(b, dtype=tf.float32, name='b')

    def f(x):
        z = tf.matmul(x, W) + b
        shape = z.shape.as_list()
        shape = [-1 if s is None else s for s in shape]
        n_conds = shape[-1]
        shape = shape[:-1] + [n_hulls, n_conds // n_hulls]
        z = tf.reshape(z, shape)
        return tf.reduce_min(tf.reduce_min(z, axis=-1), axis=-1)

    return SdfFn(f)
