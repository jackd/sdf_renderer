from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from . import camera_utils
from . import shader
from . import homogeneous


def _get_ray_endpoints(offset, directions, lengths):
    return tf.expand_dims(offset, axis=1) + \
        tf.expand_dims(lengths, axis=-1) * directions


def _dot(x, y, axis=-1):
    return tf.reduce_sum(x*y, axis=axis)


def linearize_roots(f, solution, clip_val=None, check_numerics=False):
    """
    Linearize the solution to f(x) = 0.

    Assuming fn is parameterized by some parameters theta, i.e. f(x; theta)
    and the root solution f(x; theta) = 0 implicitly defines x(theta). We can
    recover the derivatives w.r.t theta using the implicit function theorem by
    differentiating w.r.t theta.

    f(x(theta); theta) = 0
    df/dx * dx/dtheta + df/dtheta = 0
    dx/dtheta = -df/dtheta / df/dx

    Rather than assign the gradients directly, we return the solution with the
    same value and first derivatives w.r.t theta by returning

    x_linearized = m*(fx - tf.stop_gradient(fx)) + tf.stop_gradient(x)

    where fx = f(x) and m = tf.stop_gradient(-1 / df/dx) at the optimal
    solution.

    Note this assumes the roots have been found exactly. Errors in the solution
    will result in inaccuracies is the gradient, though these inaccuracies may
    be small enough so as not to affect the quality of gradient descent.

    Args:
        f: function for which it is assumed that f(solution) == 0.
        solution: root(s) of f.
        clip_val: if True, clips calculated df/dx vals to outside the range
            [-clip_val, clip_val], preventing inf values for m
            (unless df/dx == 0).
        check_numerics: if True, a `tf.check_numerics` op is added to the
            linearized gradient `m`.

    Returns:
        linearized x with same shape/value as solution, and same/approximate
            approximate first derivatives w.r.t other parameters of f.
    """
    with tf.name_scope('linearized_root'):
        sol0 = tf.stop_gradient(solution, name='s0')
        fn_val = f(sol0)

        gradient, = tf.gradients(fn_val, sol0)
        if clip_val is not None:
            # TODO: deal with when gradient == 0, hence sign(gradient) == 0
            gradient = tf.sign(gradient) * tf.maximum(
                tf.abs(gradient), clip_val)

        m = tf.stop_gradient(tf.reciprocal(-gradient), name='const_gradient')
        if check_numerics:
            m = tf.check_numerics(m, 'linearized_gradient')
        sol1 = m * (fn_val - tf.stop_gradient(fn_val)) + sol0
    return sol1


def sphere_march(
        sdf_fn, offset, directions, max_length, threshold=1e-2,
        maximum_iterations=100, back_prop=False):
    with tf.name_scope('sphere_march'):
        sdf0 = sdf_fn(offset)
        sdf0 = tf.expand_dims(sdf0, axis=-1)
        lengths = tf.tile(sdf0, (1, directions.shape[1].value))
        hit = tf.zeros(shape=tf.shape(lengths), dtype=tf.bool)
        missed = hit
        converged = hit

        def cond(lengths, hit, missed, converged):
            return tf.logical_not(tf.reduce_all(converged))

        def body(lengths, hit, missed, converged):
            points = _get_ray_endpoints(offset, directions, lengths)
            sdf = sdf_fn(points)
            lengths = lengths + sdf
            hit = tf.less_equal(sdf, threshold)
            missed = tf.greater_equal(lengths, max_length)
            converged = tf.logical_or(hit, missed)
            return lengths, hit, missed, converged

        lengths, hit, missed, converged = tf.while_loop(
            cond, body, (lengths, hit, missed, converged),
            back_prop=back_prop, maximum_iterations=maximum_iterations)

    return lengths, hit, missed


def ray_march(sdf_fn, offset, directions, max_length, n_steps=20):
    with tf.name_scope('ray_march'):
        n_rays = tf.shape(directions)[1]
        lengths0 = sdf_fn(offset)
        gap = max_length - lengths0
        step0 = gap / n_steps
        lengths0 = tf.tile(tf.expand_dims(lengths0, axis=1), (1, n_rays))
        step0 = tf.tile(tf.expand_dims(step0, axis=-1), (1, n_rays))

        offset = tf.expand_dims(offset, axis=-2)

        def cond(lengths, step):
            return True

        def body(lengths, step):
            upper = tf.expand_dims(lengths + step, axis=-1)*directions + offset
            sdf = sdf_fn(upper)
            passed = tf.less(sdf, 0)
            lengths = tf.where(passed, lengths, lengths + step)
            step = tf.where(passed, step / 2, step)
            return lengths, step

        lengths, step = tf.while_loop(
            cond, body, (lengths0, step0), back_prop=False,
            maximum_iterations=n_steps)
        missed = tf.equal(step, step0)
    return lengths, step, missed


def disection_march(
        sdf_fn, offset, directions, max_length, n_splits_initial=16,
        n_splits_loop=8, n_iterations=5):
    raise NotImplementedError()
    with tf.name_scope('disection_march'):
        d_shape = tf.shape(directions)
        batch_size = d_shape[0]
        n_rays = d_shape[1]
        n = n_splits_initial
        sdf0 = sdf_fn(offset)
        gap = max_length - sdf0
        zero = tf.zeros((), dtype=tf.float32)
        one = tf.ones((), dtype=tf.float32)
        g0 = tf.lin_space(zero, one, n)
        g0 = tf.expand_dims(g0, axis=0)
        lengths = tf.expand_dims(sdf0, axis=-1) + \
            tf.expand_dims(gap, axis=1)*g0
        grid = tf.expand_dims(tf.expand_dims(lengths, axis=-1), axis=1)
        directions = tf.expand_dims(directions, axis=-2)
        offset = tf.expand_dims(tf.expand_dims(offset, axis=-2), axis=-2)
        points = offset + directions*grid
        sdf = sdf_fn(points)
        signs = tf.sign(sdf)
        changes = tf.not_equal(signs[..., :-1], signs[..., 1:])
        ones = tf.expand_dims(
            tf.ones(shape=tf.shape(changes)[:-1], dtype=tf.bool), axis=-1)
        changes = tf.concat((changes, ones), axis=-1)
        i = tf.argmax(
            tf.cast(changes, dtype=tf.uint8), axis=-1, output_type=tf.int32)
        missed = tf.equal(i, n)
        batch_index = tf.tile(
            tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=-1),
            (1, n_rays))

        bi = tf.stack((batch_index, i), axis=-1)
        lengths = tf.gather_nd(lengths, bi)

        n = n_splits_loop
        g0 = tf.reshape(tf.lin_space(zero, one, n), (1, 1, -1))

        def cond(*args, **kwargs):
            return True

        def body(lengths, missed):
            gap = max_length - lengths
            lengths = tf.expand_dims(gap, axis=-1)
            # points = offset + lengths*directions
            # TODO: or maybe just call linear solver??
            raise NotImplementedError('TODO')

        lengths, missed = tf.while_loop(
            cond, body, (lengths,  missed), back_prop=False,
            maximum_iterations=n_iterations)
        hit = tf.logical_not(missed)

    return lengths, hit, missed


def get_intersections(
        sdf_fn, offset, directions, max_length=1, linearize=False,
        algorithm='sphere_march', **kwargs):
    """
    Get the intersection points.

    Args:
        sdf_fn: function mapping [batch_size, n_pixels, 3] coordinates to
            a conservative signed distance value, [batch_size, n_pixels]
        offset: [batch_size, 3] camera offset
        directions: [batch_size, n_pixels, 3] normalized ray directions
        max_length: scalar, maximum length of rays before assumed to have
            missed all surfaces
        linearize: if True, linearizes the solution. See `linearize_roots`
        algorithm: string indicating the intersection algorithm to use.
            See `sphere_march`, `disection_march` and `ray_march`.
        **kwargs: passed to algorithm specified by `algorithm` key.

    Returns:
        [batch_size, n_pixels, 3] 3D coordinates of intersection points of
            rays and the implicit surface defined by sdf_fn(x) == 0.
    """
    with tf.name_scope('intersections'):
        max_length = tf.convert_to_tensor(max_length, dtype=tf.float32)
        if algorithm == 'sphere_march':
            lengths, hit, missed = sphere_march(
                sdf_fn, offset, directions, max_length, **kwargs)
        elif algorithm == 'disection_march':
            lengths, hit, missed = disection_march(
                sdf_fn, offset, directions, max_length, **kwargs)
        elif algorithm == 'ray_march':
            lengths, step, missed = ray_march(
                sdf_fn, offset, directions, max_length, **kwargs)
            hit = tf.logical_not(missed)
        else:
            options = ('sphere_march', 'disection_march', 'ray_march')
            raise ValueError(
                'Unrecognized algorithm "%s". Must be one of %s'
                % (algorithm, str(options)))
        if linearize:
            def f(lengths):
                return sdf_fn(_get_ray_endpoints(offset, directions, lengths))

            lengths = linearize_roots(f, lengths)
            # lengths = fix_length_gradient(
            #     offset, directions, lengths, sdf_fn)

        lengths = tf.minimum(max_length, lengths)

        points = _get_ray_endpoints(offset, directions, lengths)

    return lengths, points, hit, missed


def get_normals(points, sdf_vals):
    with tf.name_scope('normals'):
        normals, = tf.gradients(sdf_vals, points)
        normals = normals / tf.norm(normals, axis=-1, keepdims=True)
    return normals


def render(
        sdf_fn,
        camera_position,
        camera_lookat,
        camera_up,
        image_height,
        image_width,
        fov_y,
        max_ray_length,
        surface_color_fn,
        intersection_kwargs={},
        directional_directions=None,
        directional_intensities=None,
        point_positions=None,
        point_intensities=None,
        ambient_intensity=None,
        attenuation_fn=None,
        gamma=None):
    """
    Args:
        sdf_fn: signed distance function
        camera_position: position of camera with shape [batch_size, 3] or [3]
        camera_lookat: direction camera is pointing with shape [batch_size, 3]
            or [3]
        camera_up: up direction of camera with shape [batch_size, 3]
        fov_y: float, 0D tensor, or 1D tensor with shape [batch_size]
            specifying desired output image y field of view in degrees.
        surface_color_fn: function mapping intersection points and hit bool to
            dict keyed by:
                ambient_colors
                diffuse_colors
            See `shader.get_colors` for how these are used.
        intersection_kwargs: selected kwargs for render.intersection. Any/all
            of "threshold", "back_prop", "maximum_iterations".
            See `get_intersections`
        rest: see `shader.get_colors`

    Returns:
        colors: float32 tensor of shape
            [batch_size, image_height, image_width, n_channels].
        hit: bool tensor indicating which rays hit the surface, shape
            [batch_size, image_height, image_width].
        missed: bool tensor indicating which rays missed the surface, shape
            [batch_size, image_height, image_width]. Each ray either hits,
            misses, or fails to converge.
    """
    camera_matrices = camera_utils.look_at(
        camera_position, camera_lookat, camera_up)

    camera_matrices = camera_matrices[:, :3]
    camera_rotation, camera_offset = homogeneous.split_homogeneous(
        camera_matrices)

    focal_length_px = camera_utils.get_focal_length(image_height, fov_y, 'deg')
    directions = camera_utils.get_transformed_camera_rays(
        image_height, image_width, focal_length_px, camera_rotation)

    lengths, intersections, hit, missed = get_intersections(
            sdf_fn, camera_position, directions, max_length=max_ray_length,
            **intersection_kwargs)

    normals = get_normals(intersections, sdf_fn(intersections))

    surface_colors = surface_color_fn(intersections, hit)

    pixel_colors = shader.get_colors(
        normals, intersections,
        point_positions=point_positions,
        point_intensities=point_intensities,
        ambient_intensity=ambient_intensity,
        attenuation_fn=attenuation_fn,
        **surface_colors)

    # gamma correction
    if gamma is not None:
        pixel_colors = shader.gamma_correction(pixel_colors, gamma)

    # reshaping
    px_shape = (-1, image_height, image_width)
    pixel_colors = tf.reshape(
            pixel_colors, px_shape + (pixel_colors.shape[-1].value,))
    hit = tf.reshape(hit, px_shape)
    missed = tf.reshape(missed, px_shape)

    return pixel_colors, hit, missed
