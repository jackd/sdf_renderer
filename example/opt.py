#!/usr/bin/python
"""
Basic example of optimization.

We render a scene with a known scale factor, then try to find that scale factor
by taking the original scene and minimizing pixel differences to recover the
scale factor.

Note: some combinations of hyperparameters result in the linearized solution
returning NaNs.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from sdf_renderer.sdf.primitives import Sphere, Box
from sdf_renderer.render import render


def opt_example(
        height=128, width=None, back_prop=False, linearize=True,
        threshold=1e-5, gamma=0.9, is_rgb=False, maximum_iterations=100):
    if width is None:
        width = height

    fov_y = 40.0
    max_length = 3

    # attenuation_fn = None

    n_steps = 80

    def attenuation_fn(x):
        return 1 / tf.square(x)

    sphere = Sphere(radius=0.3).translate((0.2, 0, 0))
    box = Box((0.1, 0.2, 0.3)).translate((0, -0.1, 0))
    sdf = sphere.union(box)
    scale_factor = tf.Variable(1., dtype=tf.float32)
    sdf = sdf.scale(tf.abs(scale_factor))

    eye = tf.constant([[1, 1, 1]], dtype=tf.float32)
    center = tf.zeros(shape=(1, 3), dtype=tf.float32)
    world_up = tf.constant([[0, 0, 1]], dtype=tf.float32)

    fov_y = tf.ones(shape=(1,), dtype=tf.float32) * fov_y

    intersection_kwargs = dict(
        threshold=threshold, back_prop=back_prop, linearize=linearize,
        maximum_iterations=maximum_iterations)

    if is_rgb:
        directional_directions = -tf.eye(3, batch_shape=(1,), dtype=tf.float32)
        directional_intensities = tf.eye(3, batch_shape=(1,), dtype=tf.float32)

        def color_fn(intersections, hit):
            on = tf.ones_like(intersections)
            off = tf.zeros_like(intersections)
            hit = tf.tile(tf.expand_dims(hit, axis=-1), (1, 1, 3))
            diffuse_colors = tf.where(hit, on, off)
            return dict(diffuse_colors=diffuse_colors)
    else:
        directional_directions = tf.constant(
            [[[-1, -0.75, -0.5]]], dtype=tf.float32)
        # light_directions = -tf.ones(shape=(1, 1, 3), dtype=tf.float32)
        directional_intensities = tf.ones(shape=(1, 1, 1), dtype=tf.float32)

        def color_fn(intersections, hit):
            kwargs = dict(shape=tf.shape(intersections)[:-1], dtype=tf.float32)
            on = tf.ones(**kwargs)
            off = tf.zeros(**kwargs)
            diffuse_colors = tf.expand_dims(tf.where(hit, on, off), axis=-1)
            return dict(diffuse_colors=diffuse_colors)

    point_positions = -directional_directions
    point_intensities = directional_intensities

    ambient_intensity = None

    light_kwargs = dict(
        ambient_intensity=ambient_intensity,
        directional_directions=directional_directions,
        directional_intensities=directional_intensities,
        point_positions=point_positions,
        point_intensities=point_intensities
    )

    pixel_colors, hit_px, missed_px = render(
        sdf, eye, center, world_up, height, width, fov_y,
        max_length, color_fn, intersection_kwargs,
        attenuation_fn=attenuation_fn, gamma=gamma, **light_kwargs)

    if not is_rgb:
        pixel_colors = tf.squeeze(pixel_colors, axis=-1)

    with tf.Session() as sess:
        sess.run(scale_factor.initializer)
        c_target, hp_target, mp_target = sess.run(
            (pixel_colors, hit_px, missed_px),
            feed_dict={scale_factor: 0.75})

    loss = tf.reduce_mean(tf.square(pixel_colors - c_target))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
    optimizer = tf.train.MomentumOptimizer(learning_rate=2e-1, momentum=0.75)
    opt = optimizer.minimize(loss)
    scale_grad, = tf.gradients(loss, scale_factor)

    # with tf.control_dependencies([tf.add_check_numerics_ops()]):
    #     opt = tf.identity(opt)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t = time.time()
        for i in range(n_steps):
            lv, _ = sess.run((loss, opt))
            print(lv)
        dt = time.time() - t
        print('dt = %.2f' % dt)
        print(sess.run(scale_factor))
        c, hp, mp = sess.run((pixel_colors, hit_px, missed_px))

    def vis(
            pixel_colors, hit_pixels, missed_pixels, color_target,
            hit_target, missed_target):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(pixel_colors, cmap=None if is_rgb else 'gray')
        plt.figure()
        plt.imshow(color_target, cmap=None if is_rgb else 'gray')
        plt.show()

    vis(c[0], hp[0], mp[0], c_target[0], hp_target[0], mp_target[0])


# opt_example(linearize=False, back_prop=True, is_rgb=True)
# opt_example(threshold=1e-7, maximum_iterations=200, is_rgb=True)
opt_example()
