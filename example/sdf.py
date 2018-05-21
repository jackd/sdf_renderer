#!/usr/bin/python
"""Script for visualizing SDF functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
# from sdf_renderer.sdf.primitives import Sphere
from sdf_renderer.sdf.smooth import SmoothBox
from sdf_renderer.sdf.primitives import Box


n = 64
# sphere = Sphere(radius=0.3).translate((0.2, 0, 0))
# box = Box((0.1, 0.2, 0.3))
b1 = SmoothBox((0.1, 0.2, 0.3))
b2 = Box((0.1, 0.2, 0.3))
# box = SmoothBox((0.1, 0.2, 0.3), smoother='power', k=8)

b1 = b1.translate((0, -0.1, 0))
b2 = b2.translate((-0.1, 0.1, 0.1))
sdf = b1.union(b2)
# sdf = sphere.union(box)
# sdf = box
nj = n*1j
coords = np.mgrid[-0.5:0.5:nj, -0.5:0.5:nj, -0.5:0.5:nj] + 1e-3
coords = np.transpose(coords, (1, 2, 3, 0))
coords_tf = tf.constant(coords, dtype=tf.float32)
sdf_vals = sdf(coords_tf)

with tf.Session() as sess:
    s = sess.run(sdf_vals)


def vis(s, coords):
    import sdf_renderer.vis as vis
    vis.vis_contours(s, coords)
    vis.vis_axes()
    vis.show()


vis(s, coords)
