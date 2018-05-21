#!/usr/bin/python
"""Script for visualizing camera ray/SDF intersections."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sdf_renderer.sdf.primitives import Sphere, Box
import sdf_renderer.camera_utils as camera_utils
from sdf_renderer.render import get_intersections, get_normals


n = 64
max_length = 3
image_height = 64
image_width = 64

# kwargs = dict(algorithm='sphere_march', threshold=1e-2, back_prop=False)
# kwargs = dict(algorithm='disection_march')
kwargs = dict(algorithm='ray_march')

linearize = True


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
fov_y = tf.ones(shape=(1,), dtype=tf.float32) * 40.0

eye = tf.concat([eye, eye], axis=0)
center = tf.concat([center, center], axis=0)
world_up = tf.concat([world_up, world_up], axis=0)
aspect_ratio = tf.concat([aspect_ratio, aspect_ratio], axis=0)
fov_y = tf.concat([fov_y, fov_y], axis=0)

camera_matrices = camera_utils.look_at(eye, center, world_up)
R, t = camera_utils.split_homogeneous(camera_matrices)
directions = camera_utils.get_transformed_camera_rays(
    image_height, image_width, fov_y, R)

lengths, intersections, hit, missed = get_intersections(
        sdf, eye, directions, max_length=max_length, linearize=linearize,
        **kwargs)

normals = get_normals(intersections, sdf(intersections))


with tf.Session() as sess:
    i, d, inter, n, h, m, s = sess.run(
        (eye, directions, intersections, normals, hit, missed, sdf_vals))


def vis(camera_location, ray_directions, intersections, normals, hit, missed,
        sdf_vals, coords):
    import sdf_renderer.vis as vis
    vis.vis_rays(camera_location, ray_directions)
    vis.vis_points(
        intersections[hit], scale_factor=0.02, color=(0, 1, 0))
    vis.vis_points(
        intersections[missed],
        scale_factor=0.02, color=(0, 0, 1))
    vis.vis_points(
        intersections[np.logical_not(np.logical_or(hit, missed))],
        scale_factor=0.02, color=(1, 0, 0))

    vis.vis_normals(
        intersections[hit], normals[hit], color=(0, 0, 0), opacity=0.2)
    vis.vis_axes()
    vis.vis_contours(sdf_vals, coords)
    vis.show()


i = i[0]
d = d[0]
inter = inter[0]
h = h[0]
m = m[0]
n = n[0]
vis(i, d, inter, n, h, m, s, coords)
