"""
Differentiable shading.

Aspects taken from tf_mesh_renderer.

https://github.com/google/tf_mesh_renderer
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_colors(
        normals=None, positions=None,
        ambient_colors=None, diffuse_colors=None,
        ambient_intensity=None,
        directional_directions=None, directional_intensities=None,
        point_positions=None, point_intensities=None, attenuation_fn=None):
    """
    Wrapper method that combines ambient, point and directional light sources.

    Note: no gamma correction is applied. Consider following with
    `gamma_correction`.

    Args:
        normals: [batch_size, n_pixels, 3] 3D normals of surface intersection
            points
        positions: [batch_size, n_pixels, 3] 3D coordinates of surface
            intersection points
        ambient_colors: [batch_size, n_pixels, n_channels] ambient colors of
            intersection points
        directional_directions: [batch_size, n_directional_lights, 3] 3D
            directional vector of directional lights
        directional_intensities: [batch_size, n_directional_lights, n_channels]
            intensities of directional lights
        point_positions: [batch_size, n_point_lights, 3] 3D positions of
            point light sources
        point_intensitites: [batch_size, n_point_lights, n_channels] light
            intensities of each poitn light source
        attenuation_fn: Function producing an attenuation factor based on
            distance from point light sources to intersection points

    Returns:
        [batch_size, n_pixels, n_channels] dynamic pixel intensities. The sum
        of ambient colors, diffuse colors from point lights and diffuse colors
        from directional lights.

    See also:
        get_ambient_colors
        get_point_diffuse_colors
        get_directional_diffuse_colors
    """
    colors = []
    if ambient_intensity is not None:
        if ambient_colors is None:
            raise ValueError('ambient_colors required if ambient_intensity is')
        colors.append(get_ambient_colors(ambient_colors, ambient_intensity))
    if directional_directions is not None:
        assert(diffuse_colors is not None)
        assert(directional_intensities is not None)
        colors.append(get_directional_diffuse_colors(
            normals, diffuse_colors,
            directional_directions, directional_intensities))
    if point_positions is not None:
        assert(point_intensities is not None)
        colors.append(get_point_diffuse_colors(
            positions, normals, diffuse_colors,
            point_positions, point_intensities,
            attenuation_fn))
    if len(colors) == 0:
        raise ValueError('No lighting information provided')
    if len(colors) == 1:
        color = colors[0]
    else:
        color = tf.add_n(colors)
    return color


def get_ambient_colors(ambient_colors, ambient_intensity):
    """
    Get the ambient color of a scene.

    Args:
        ambient_color: [batch_size, n_pixels, n_channels] if
            grayscale. Ambient color of each intersection point.
        ambient_intensity: [batch_size, n_channels] ambient light intensity of
            the scene.

    Returns:
        combined ambient color.
    """
    return ambient_colors * tf.expand_dims(ambient_intensity, axis=1)


def get_point_diffuse_colors(
        positions, normals, diffuse_colors,
        light_positions, light_intensities, attenuation_fn=None):
    """
    Get diffuse colors from point light sources.

    Args:
        positions: [batch_size, n_pixels, 3] positions of intersections of
            camera rays and surface
        normals: [batch_size, n_pixels, 3] surface normals
        diffuse_colors: [batch_size, n_pixels, 3] or [batch_size, n_pixels]
            if grayscale. Static diffuse colors of intersection points
        light_positions: [batch_size, n_lights, 3] 3D coordinates of lights
        light_intensities: [batch_size, n_lights, n_channels] light intensities
            of each point light source
        attenuation_fn: optional function that maps the distance between each
            ligth/intersection to an attenuation factor,
            attenuated_intensity = intensity * attenuation_fn(distance)

    Returns:
        Dynamic diffuse colors, same shape as input `diffuse_colors`.
    """
    n_pixels = normals.shape[1].value
    if len(normals.shape) != 3 or normals.shape[2].value != 3:
        raise ValueError('normals must have shape [batch_size, n_pixels, 3]')
    if len(positions.shape) != 3 or positions.shape[2].value != 3:
        raise ValueError('positions must have shape [batch_size, n_pixels, 3]')
    if diffuse_colors.shape[1].value != n_pixels:
        raise ValueError(
            'diffuse_colors must have same dimension 1 (n_pixels) as normals')
    if positions.shape[1].value != n_pixels:
        raise ValueError(
            'positions must have same dimension 1 (n_pixels) as normals')
    n_lights = light_positions.shape[1].value
    if light_intensities.shape[1].value != n_lights:
        raise ValueError(
            'light_intensities must have same dimension 1 as light_positions')
    if len(light_positions.shape) != 3 or light_positions.shape[2].value != 3:
        raise ValueError(
            'light_positions must have shape [batch_size, n_lights, 3]')
    if len(light_intensities.shape) != 3:
        raise ValueError('light_intensities must have rank 3')
    if len(diffuse_colors.shape) != 3:
        raise ValueError('diffuse_colors must have rank 3')

    with tf.name_scope('point_diffuse_colors'):
        positions = tf.expand_dims(positions, axis=-2)
        light_positions = tf.expand_dims(light_positions, axis=-3)
        light_offset = positions - light_positions
        dists = tf.norm(light_offset, axis=-1, keepdims=True)
        light_directions = light_offset / dists
        normals = tf.expand_dims(normals, axis=-2)
        dots = tf.nn.relu(-tf.reduce_sum(normals * light_directions, axis=-1))
        light_intensities = tf.expand_dims(light_intensities, axis=1)
        if attenuation_fn is not None:
            attenuation = attenuation_fn(tf.squeeze(dists, axis=-1))
            dots = dots * attenuation
            dots = tf.expand_dims(dots, axis=-1)
        color = tf.reduce_sum(dots * light_intensities, axis=2)
    return color


def get_directional_diffuse_colors(
        normals, diffuse_colors, light_directions, light_intensities):
    """
    Get diffuse colors from directional light sources.

    Args:
        normals: [batch_size, n_pixels, 3]
        diffuse_colors: [batch_size, n_pixels, n_channels] static diffuse color
        light_direcitons: [batch_size, n_lights, 3] direction of light.
        light_intensities: [batch_size, n_lights, n_channels] intensities of
            each directional light.

    Returns:
        [batch_size, n_pixels, n_channels] dynamic diffuse color.
    """
    n_pixels = normals.shape[1].value
    if len(normals.shape) != 3 or normals.shape[-1].value != 3:
        raise ValueError('normals must have shape [batch_size, n_pixels, 3], '
                         'got %s' % str(normals.shape.as_list()))
    if diffuse_colors.shape[1].value != n_pixels:
        raise ValueError(
            'diffuse_colors.shape[1] must be same as normals.shape[1]')
    n_lights = light_directions.shape[1]
    if len(light_directions.shape) != 3 or \
            light_directions.shape[-1].value != 3:
        raise ValueError(
            'light_directions must have shape [batch_size, n_pixels, 3], '
            'got %s' % str(normals.shape.as_list()))
    if light_intensities.shape[1].value != n_lights:
        raise ValueError(
            'light_intensities.shape[1] must be same as '
            'light_directions.shape[1]')
    if len(diffuse_colors.shape) != 3:
        raise ValueError('diffuse_colors must be rank 3')
    if len(light_intensities.shape) != 3:
        raise ValueError('light_intensities must be rank 3')

    with tf.name_scope('directional_diffuse_colors'):
        dots = tf.nn.relu(
            -tf.einsum('ijk,ilk->ijl', normals, light_directions))
        light_intensities = tf.expand_dims(light_intensities, axis=1)
        dots = tf.expand_dims(dots, axis=-1)
        color = tf.reduce_sum(dots * light_intensities, axis=2)

        return color * diffuse_colors


def gamma_correction(images, gamma, eps_offset=1e-7, eps_scale=1e-7):
    """
    Applies gamma correction to the input images.

    Tone maps the input image batch in order to make scenes with a high dynamic
    range viewable. The gamma correction factor is computed separately per
    image, but is shared between all provided channels. The exact function
    computed is:

    image_out = A*((image_in + eps_offset) - eps_offset^gamma),
    where A is an element-wide constant computed so that the maximum image
    value is approximately 1. The correction is applied to all channels, then
    clipped to exactly [0, 1].

    Note: the input image does not necessarily need to be rank 3 or 4. All
    dimensions beyond the first are treated identically.

    Args:
        images: N-D float32 tensor, N > 1. First dimension is the batch.
            The batch of images to tone map.
        gamma: 0-D float32 nonnegative tensor. Values of gamma below one
            compress relative contrast in the image, and values above one
            increase it. A value of 1 is equivalent to scaling the image to
            have a maximum value of 1.
        eps_offset: small offset applied to colors to ensure differentiable
            at zero.
        eps_scale: lower bound on image scaling to prevent division by zero
            when all pixels are low.
            image_out = image / max(max(image), image_scale).
    Returns:
        Contains the gamma-corrected colors, clipped to the range [0, 1]. Same
        shape as images
    """
    with tf.name_scope('gamma_correction'):
        if eps_offset == 0:
            images = images ** gamma
        else:
            images = \
                (images + eps_offset) ** gamma - eps_offset ** gamma

        max_colors = tf.reduce_max(
            images, axis=range(1, len(images.shape)),
            keepdims=True)
        if eps_scale != 0:
            max_colors = tf.maximum(max_colors, eps_scale)
        images = images / max_colors
        images = tf.clip_by_value(images, 0.0, 1.0)
    return images
