#!/usr/bin/python
"""Script for rendering a basic SDF scene."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def render_example(
        height=512, width=None, is_rgb=True, threshold=1e-5, gamma=0.9,
        linearize_intersections=True, back_prop=False):
    import tensorflow as tf
    # import numpy as np
    from sdf_renderer.sdf.primitives import Sphere, Box
    from sdf_renderer.render import render
    if width is None:
        width = height

    fov_y = 40.0
    max_length = 3

    def attenuation_fn(x):
        return 1 / tf.square(x)

    sphere = Sphere(radius=0.3).translate((0.2, 0, 0))
    box = Box((0.1, 0.2, 0.3)).translate((0, -0.1, 0))
    sdf = sphere.union(box)

    eye = tf.constant([[1, 1, 1]], dtype=tf.float32)
    center = tf.zeros(shape=(1, 3), dtype=tf.float32)
    world_up = tf.constant([[0, 0, 1]], dtype=tf.float32)

    fov_y = tf.ones(shape=(1,), dtype=tf.float32) * fov_y

    intersection_kwargs = dict(
        threshold=threshold, linearize=linearize_intersections,
        back_prop=back_prop)

    if is_rgb:
        directional_directions = -tf.eye(3, batch_shape=(1,), dtype=tf.float32)
        directional_intensities = tf.eye(3, batch_shape=(1,), dtype=tf.float32)

        def color_fn(intersections, hit):
            on = tf.ones_like(intersections)
            off = tf.zeros_like(intersections)
            hit = tf.expand_dims(hit, axis=-1)
            hit = tf.tile(hit, (1, 1, 3))
            diffuse_colors = tf.where(hit, on, off)
            return dict(diffuse_colors=diffuse_colors, ambient_colors=None)
    else:
        directional_directions = tf.constant(
            [[[-1, -0.75, -0.5]]], dtype=tf.float32)
        directional_intensities = tf.ones(shape=(1, 1, 1), dtype=tf.float32)

        def color_fn(intersections, hit):
            kwargs = dict(shape=tf.shape(intersections)[:-1], dtype=tf.float32)
            on = tf.ones(**kwargs)
            off = tf.zeros(**kwargs)
            diffuse_colors = tf.expand_dims(tf.where(hit, on, off), axis=-1)
            return dict(diffuse_colors=diffuse_colors, ambient_colors=None)
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
        attenuation_fn=attenuation_fn,
        gamma=gamma, **light_kwargs)

    if not is_rgb:
        pixel_colors = tf.squeeze(pixel_colors, axis=-1)

    with tf.Session() as sess:
        c, hp, mp = sess.run((pixel_colors, hit_px, missed_px))

    def vis(pixel_colors, hit_pixels, missed_pixels):
        import matplotlib.pyplot as plt
        # pixel_colors[np.logical_not(hit_pixels)] = 0
        plt.imshow(pixel_colors, cmap=None if is_rgb else 'gray')
        plt.show()

    vis(c[0], hp[0], mp[0])


render_example()
