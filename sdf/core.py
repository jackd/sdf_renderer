from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Sdf(object):
    def __call__(self, x):
        raise NotImplementedError()

    def translate(self, offset):
        return TranslatedSdf(self, offset)

    def rotate(self, rotation_matrix, is_transposed=False):
        return RotatedSdf(self, rotation_matrix, is_transposed=False)

    def scale(self, scale_factor):
        return ScaledSdf(self, scale_factor)

    def union(self, other):
        return SdfUnion(self, other)

    def intersection(self, other):
        return SdfIntersection(self, other)

    def complement(self):
        return SdfComplement(self)


class TransformedSdf(Sdf):
    def __init__(self, base):
        self._base = base

    @property
    def base(self):
        return self._base


class TranslatedSdf(TransformedSdf):
    def __init__(self, base, offset):
        self._offset = tf.convert_to_tensor(
            offset, dtype=tf.float32, name='offset')
        super(TranslatedSdf, self).__init__(base)

    @property
    def offset(self):
        return self._offset

    def __call__(self, x):
        return self._base(x - self.offset)

    def translate(self, offset):
        return self.base.translate(self.offset + offset)


class RotatedSdf(TransformedSdf):
    def __init__(self, base, rotation):
        self._rotation = tf.convert_to_tensor(
            rotation, dtype=tf.float32, name='rotation')
        super(RotatedSdf, self).__init__(base)

    @property
    def rotation(self):
        return self._rotation

    def __call__(self, x):
        rotation = self.rotation
        rotation_matrix = rotation.rotation_matrix
        is_transposed = rotation.is_transposed
        x = tf.matmul(x, rotation_matrix, transpose_b=not is_transposed)
        return self.base(x)


class ScaledSdf(TransformedSdf):
    def __init__(self, base, scale_factor):
        self._scale_factor = tf.convert_to_tensor(
            scale_factor, dtype=tf.float32, name='scale_factor')
        super(ScaledSdf, self).__init__(base)

    @property
    def scale_factor(self):
        return self._scale_factor

    def __call__(self, x):
        scale_factor = self.scale_factor
        return self.base(x / scale_factor) * scale_factor

    def scale(self, scale_factor):
        return self.base.scale(self.scale_factor * scale_factor)


class SdfComplement(TransformedSdf):
    def __call__(self, x):
        return tf.negative(self.base(x))

    def complement(self):
        return self.base


class SdfNaryOp(Sdf):
    def __init__(self, *operands):
        self._operands = tuple(operands)

    @property
    def operands(self):
        return self._operands


class SdfUnion(SdfNaryOp):
    def __call__(self, x):
        vals = tf.stack(tuple(sdf(x) for sdf in self.operands), axis=-1)
        return tf.reduce_min(vals, axis=-1)


class SdfIntersection(SdfNaryOp):
    def __call__(self, x):
        vals = tf.stack(tuple(sdf(x) for sdf in self.operands), axis=-1)
        return tf.reduce_max(vals, axis=-1)
