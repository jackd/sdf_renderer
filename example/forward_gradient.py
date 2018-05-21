#!/usr/bin/python
"""Script for visualizing the forward gradient."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sdf_renderer.sdf.primitives import Sphere, Box
from sdf_renderer.render import render


def forward_gradients(ys, xs, d_xs=None):
    """
    Forward-mode pushforward analogous to the pullback defined by tf.gradients.

    With tf.gradients, grad_ys is the vector being pulled back, and here d_xs
    is the vector being pushed forward.

    See https://github.com/renmengye/tensorflow-forward-ad/issues/2
    """
    if isinstance(ys, (list, tuple)):
        v = tuple(tf.zeros_like(y) for y in ys)
    else:
        v = tf.zeros_like(ys)
    g = tf.gradients(ys, xs, grad_ys=v)
    return tf.gradients(g, v, grad_ys=d_xs)


n = 64
# threshold = 1e-2
# threshold = 1e-7
threshold = 1e-5
# threshold = 0

image_height = 512
image_width = 512
fov_y = 40.0
max_length = 3

is_rgb = False
# is_rgb = True

# gamma = 1.0001
# gamma = 1
# gamma = 0.99999
gamma = 0.9
# gamma = 1 / 2.5
# gamma = None


sphere = Sphere(radius=0.3).translate((0.2, 0, 0))
box = Box((0.1, 0.2, 0.3)).translate((0, -0.1, 0))
sdf = sphere.union(box)
scale_factor = tf.Variable(1., dtype=tf.float32)
sdf = sdf.scale(scale_factor)

eye = tf.constant([[1, 1, 1]], dtype=tf.float32)
center = tf.zeros(shape=(1, 3), dtype=tf.float32)
world_up = tf.constant([[0, 0, 1]], dtype=tf.float32)

fov_y = tf.ones(shape=(1,), dtype=tf.float32) * fov_y
light_positions = tf.constant(
    [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=tf.float32)
light_intensities = tf.ones(shape=(1, 3, 3), dtype=tf.float32)

intersection_kwargs = dict(threshold=threshold, back_prop=True, linearize=True)


if is_rgb:
    directional_directions = -tf.eye(3, batch_shape=(1,), dtype=tf.float32)
    directional_intensities = tf.eye(3, batch_shape=(1,), dtype=tf.float32)

    def color_fn(intersections, hit):
        diffuse_colors = tf.ones_like(intersections)
        return dict(diffuse_colors=diffuse_colors)
else:
    directional_directions = tf.constant(
        [[[-1, -0.75, -0.5]]], dtype=tf.float32)
    # light_directions = -tf.ones(shape=(1, 1, 3), dtype=tf.float32)
    directional_intensities = tf.ones(shape=(1, 1, 1), dtype=tf.float32)
    diffuse_colors = tf.ones(shape=(1, image_height*image_width, 1))

    def color_fn(intersections, hit):
        diffuse_colors = tf.ones(
            shape=tf.shape(intersections)[:-1], dtype=tf.float32)
        return dict(diffuse_colors=tf.expand_dims(diffuse_colors, axis=-1))
point_positions = -directional_directions
point_intensities = directional_intensities

ambient_color = None
ambient_intensity = None

light_kwargs = dict(
    ambient_intensity=ambient_intensity,
    directional_directions=directional_directions,
    directional_intensities=directional_intensities,
    point_positions=point_positions,
    point_intensities=point_intensities
)

pixel_colors, hit_px, missed_px = render(
    sdf, eye, center, world_up, image_height, image_width, fov_y,
    max_length, color_fn, intersection_kwargs,
    attenuation_fn=lambda x: 1/tf.square(x),
    gamma=gamma, **light_kwargs)

if not is_rgb:
    pixel_colors = tf.squeeze(pixel_colors, axis=-1)

scale_grad, = forward_gradients(pixel_colors, scale_factor)
scale_grad = tf.clip_by_value(scale_grad, -1, 1)


with tf.Session() as sess:
    sess.run(scale_factor.initializer)
    c, hp, mp, sg = sess.run(
        (pixel_colors, hit_px, missed_px, scale_grad))


def vis(pixel_colors, hit, missed_pixels, pixel_grad):
    import matplotlib.pyplot as plt
    # ph = pixel_grad[hit]
    # print(np.min(ph), np.max(ph))
    pixel_grad -= np.min(pixel_grad[hit])
    sf = np.max(pixel_grad[hit])
    if sf > 0:
        pixel_grad /= sf
    pixel_grad[np.logical_not(hit)] = -1
    pixel_grad += 1
    plt.subplot(131)
    plt.title('rendering')
    plt.imshow(pixel_colors, cmap=None if is_rgb else 'gray')
    plt.subplot(132)
    plt.title('forward grad')
    plt.imshow(pixel_grad, cmap=None if is_rgb else 'gray')
    plt.subplot(133)
    plt.title('hit')
    plt.imshow(hit, cmap='gray')
    plt.show()


vis(c[0], hp[0], mp[0], sg[0])
