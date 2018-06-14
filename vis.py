"""Visualization functions using mayavi."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mayavi import mlab
import numpy as np


def vis_axes(length=1):
    mlab.quiver3d([0], [0], [0], [length], [0], [0], color=(1, 0, 0))
    mlab.quiver3d([0], [0], [0], [0], [length], [0], color=(0, 1, 0))
    mlab.quiver3d([0], [0], [0], [0], [0], [length], color=(0, 0, 1))


def vis_origin():
    vis_points([0, 0, 0], color=(0, 0, 0), scale_factor=0.2)


def vis_points(p, **kwargs):
    if p.shape == (3,):
        p = np.expand_dims(p, axis=0)
    mlab.points3d(*p.T, **kwargs)


def vis_normals(points, normals, **kwargs):
    x, y, z = points.T
    u, v, w = normals.T
    mlab.quiver3d(x, y, z, u, v, w, **kwargs)


def vis_contours(sdf_vals, coords, contours=[0]):
    x, y, z = (coords[..., i] for i in range(3))
    mlab.contour3d(x, y, z, sdf_vals, contours=contours, transparent=True)


def vis_rays(offset, directions, **kwargs):
    """
    Based on mayavi doc.

    http://docs.enthought.com/mayavi/mayavi/auto/example_plotting_many_lines.html
    """
    start = np.tile(np.expand_dims(
        offset, axis=0), (directions.shape[0], 1))
    end = start + directions
    start_end = np.stack((start, end), axis=1)
    connections = []
    x = []
    y = []
    z = []
    index = 0
    for se in start_end:
        x.append(se[:, 0])
        y.append(se[:, 1])
        z.append(se[:, 2])
        N = len(se)
        connections.append(np.vstack(
                   [np.arange(index,   index + N - 1.5),
                    np.arange(index + 1, index + N - .5)]).T)
        index += N

    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    connections = np.vstack(connections)
    # Create the points
    src = mlab.pipeline.scalar_scatter(x, y, z)

    # Connect them
    src.mlab_source.dataset.lines = connections
    src.update()

    # The stripper filter cleans up connected lines
    lines = mlab.pipeline.stripper(src)

    # Finally, display the set of lines
    mlab.pipeline.surface(
        lines, line_width=0.2, opacity=.4, **kwargs)


show = mlab.show
figure = mlab.figure
