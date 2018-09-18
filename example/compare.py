#!/usr/bin/python
"""Script for visualizing scene in mayavi + rendered image."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sdf_renderer.sdf.primitives import Sphere, Box
import sdf_renderer.camera_utils as camera_utils
import sdf_renderer.homogeneous as homogeneous
from sdf_renderer.render import get_intersections, get_normals, render

n = 64
# threshold = 1e-1
threshold = 1e-2
# threshold = 1e-5
# threshold = 1e-7
# threshold = 0

image_height = 64
image_width = 64
fov_y = 40.
max_length = 3

is_rgb = False
# is_rgb = True

fix_lengths = True
# fix_lengths = False

# gamma = 0.9
gamma = None


sphere = Sphere(radius=0.3).translate((0.2, 0, 0))
box = Box((0.1, 0.2, 0.3)).translate((0, -0.1, 0))
sdf = sphere.union(box)

nj = n*1j
coords = np.mgrid[-0.5:0.5:nj, -0.5:0.5:nj, -0.5:0.5:nj]
coords = np.transpose(coords, (1, 2, 3, 0))
coords_tf = tf.constant(coords, dtype=tf.float32)
sdf_vals = sdf(coords_tf)

eye = tf.constant([[1, 1, 1]], dtype=tf.float32)
center = tf.zeros(shape=(1, 3), dtype=tf.float32)
world_up = tf.constant([[0, 0, 1]], dtype=tf.float32)

aspect_ratio = tf.ones(shape=(1,), dtype=tf.float32)
fov_y = tf.ones(shape=(1,), dtype=tf.float32) * fov_y
light_positions = tf.constant(
    [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=tf.float32)
light_intensities = tf.ones(shape=(1, 3, 3), dtype=tf.float32)

camera_matrices = camera_utils.look_at(eye, center, world_up)
R, t = homogeneous.split_homogeneous(camera_matrices)
directions = camera_utils.get_transformed_camera_rays(
    image_height, image_width, fov_y, R)

intersection_kwargs = dict(
    threshold=threshold, back_prop=False, linearize=fix_lengths)
lengths, intersections, hit, missed = get_intersections(
        sdf, eye, directions, max_length=max_length, **intersection_kwargs)

normals = get_normals(intersections, sdf(intersections))

if is_rgb:
    directional_directions = -tf.eye(3, batch_shape=(1,), dtype=tf.float32)
    directional_intensities = tf.eye(3, batch_shape=(1,), dtype=tf.float32)

    def color_fn(intersections, hit):
        on = tf.ones_like(intersections)
        off = tf.zeros_like(intersections)
        diffuse_colors = tf.where(hit, on, off)
        return dict(diffuse_colors=diffuse_colors)
else:
    directional_directions = tf.constant(
        [[[-1, -0.75, -0.5]]], dtype=tf.float32)
    # light_directions = -tf.ones(shape=(1, 1, 3), dtype=tf.float32)
    directional_intensities = tf.ones(shape=(1, 1, 1), dtype=tf.float32)
    diffuse_colors = tf.ones(shape=(1, image_height*image_width, 1))

    def color_fn(intersections, hit):
        kwargs = dict(shape=tf.shape(intersections)[:-1], dtype=tf.float32)
        on = tf.ones(**kwargs)
        off = tf.zeros(**kwargs)
        diffuse_colors = tf.where(hit, on, off)
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


with tf.Session() as sess:
    i, d, inter, n, h, m, s, c, hp, mp = sess.run(
        (eye, directions, intersections, normals, hit, missed, sdf_vals,
         pixel_colors, hit_px, missed_px))


def vis(camera_location, ray_directions, intersections, normals, hit, missed,
        sdf_vals, coords, pixel_colors, hit_pixels, missed_pixels):
    import sdf_renderer.vis as vis
    # import matplotlib.pyplot as plt
    # pixel_colors[np.logical_not(hit_pixels)] = 0
    # plt.imshow(pixel_colors, cmap=None if is_rgb else 'gray')
    from PIL import Image
    if not is_rgb:
        pixel_colors = np.stack([pixel_colors]*3, axis=-1)
    pixel_colors[pixel_colors > 1] = 1
    pixel_colors[pixel_colors < 0] = 0
    # pixel_colors /= np.max(pixel_colors)
    pixel_colors = (pixel_colors*255).astype(np.uint8)
    Image.fromarray(pixel_colors).resize((255, 255)).show()

    # plt.show(block=False)
    print(camera_location.shape)
    print(ray_directions.shape)
    vis.vis_rays(camera_location, ray_directions[::8])
    vis.vis_points(
        intersections[hit], scale_factor=0.02, color=(0, 1, 0))
    vis.vis_points(
        intersections[missed],
        scale_factor=0.02, color=(0, 0, 1))

    vis.vis_points(
        intersections[np.logical_not(np.logical_or(hit, missed))],
        scale_factor=0.02, color=(1, 0, 0))
    vis.vis_normals(
        intersections[hit], normals[hit], color=(0, 0, 0))
    vis.vis_axes()
    vis.vis_contours(sdf_vals, coords)
    vis.show()
    # plt.close()


i = i[0]
d = d[0]
inter = inter[0]
h = h[0]
m = m[0]
n = n[0]
c = c[0]
hp = hp[0]
mp = mp[0]
vis(i, d, inter, n, h, m, s, coords, c, hp, mp)
