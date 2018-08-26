import traceback
import sys
import functools

import numpy as np
from scipy import ndimage

import ndarray_tools
from mappables import Pixel, Shape


PIXEL_VALUE_TO_COLOR_DICT = {0: "Black", 1: "White"}


@property  # TODO?
def should_recalculate_context_strict(*_, changes_cache={}, **__):
    return True


@property  # TODO.
def should_recalculate_context_lax(*, changes_cache={}):
    return False


RESIZE_STRICTNESS = "STRICT"
RESIZE_STRICTNESS_DICT = {
    "STRICT": should_recalculate_context_strict,
    "LAX": should_recalculate_context_lax
}


@property
def should_recalculate():
    return RESIZE_STRICTNESS_DICT.get(RESIZE_STRICTNESS)


def quad_neighbor_pixels_from_ndarray(image, *, pixel_class=None):
    pixels = pixels_from_ndarray(image, pixel_class=pixel_class)
    pixel_quad_neighbors_from_ndarray(pixels)
    return pixels


def pixels_from_ndarray(image, *, pixel_class=None):
    if pixel_class is None:
        pixel_class = Pixel
    pixel_data_type = np.dtype(pixel_class)
    vertical_indices, horizontal_indices = np.indices(image.shape)
    return _iter_convert_pixels(
        image, vertical_indices, horizontal_indices, pixel_data_type,
        lambda pixel_value, vertical_index, horizontal_index:
        pixel_class(pixel_value, (vertical_index, horizontal_index))
    )


def _iter_convert_pixels(image_to_iter, vertical_indices, horizontal_indices, pixel_data_type, pixel_from_attributes):
    # out = np.empty_like(image_to_iter, dtype=out_type)
    pixel_convert = np.frompyfunc(pixel_from_attributes, 3, 1)
    convert_iterable = np.nditer([image_to_iter, vertical_indices, horizontal_indices, None],
                                 flags=['external_loop', 'buffered', 'refs_ok'],
                                 op_dtypes=['uint8', 'int', 'int', pixel_data_type],
                                 op_flags=[['readonly'], ['readonly'], ['readonly'], ['writeonly', 'allocate']])
    for pixel_value, vertical_location, horizontal_location, this_output in convert_iterable:
        this_output[...] = pixel_convert(pixel_value, vertical_location, horizontal_location)
    return convert_iterable.operands[3]


def pixel_quad_neighbors_from_ndarray(pixel_ndarray):
    _assign_ndarray_quad_neighbors(pixel_ndarray, *_get_ndarray_quad_neighbors(pixel_ndarray))


def _get_ndarray_quad_neighbors(pixel_ndarray):
    up_neighbors = ndarray_tools.np_shift_v(pixel_ndarray, 1)
    left_neighbors = ndarray_tools.np_shift_h(pixel_ndarray, 1)
    down_neighbors = ndarray_tools.np_shift_v(pixel_ndarray, -1)
    right_neighbors = ndarray_tools.np_shift_h(pixel_ndarray, -1)
    return up_neighbors, left_neighbors, down_neighbors, right_neighbors


def _assign_ndarray_quad_neighbors(pixel_ndarray, up_neighbors, left_neighbors, down_neighbors, right_neighbors):
    np_set_neighbors = np.frompyfunc(_set_pixel_neighbors, 5, 0)
    np_set_neighbors(pixel_ndarray, up_neighbors, left_neighbors, down_neighbors, right_neighbors)


def _set_pixel_neighbors(pix, *neighbors):
    """
    Create a tuple of references to neighbors in pix.

    :Parameters:
        :param Pixel pix: Pixel object to set neighbor references within.
        :param Pixels tuple neighbors: The (up, left, down, right) Pixel neighbors of pix.
    :rtype: None
    :return: None
    """
    pix.neighbors = neighbors


def binary_label_ndarray(image):
    # White Shapes label
    white_labeled, white_labels = label_ndarray_ones(image)

    # Black Shapes label
    black_labeled, black_labels = label_ndarray_ones(1 - image)

    return black_labeled, black_labels, white_labeled, white_labels


def label_ndarray_ones(image, *np_args, **np_kwargs):
    """
    Segment and label the regions of connected 1's in an image.

    :Parameters:
        :param numpy.ndarray image: The image to be labeled.
        :param np_args: (Optional) Positional arguments provided to ndimage.label after image.
        :param np_kwargs: (Optional) Keyword arguments provided to ndimage.label after np_args.
    :rtype: numpy.ndarray, numpy.ndarray
    :return: image array with ones converted to the label group id, numpy.ndarray with a label id value per index.
    """
    image_labeled, number_labels = ndimage.label(image, *np_args, **np_kwargs)
    labels = np.arange(1, number_labels + 1)
    return image_labeled, labels


def binary_shapes_from_labels(pixels,
                              black_labeled, black_labels,
                              white_labeled, white_labels,
                              *args, shape_class=None, shape_data_type=None, **kwargs):
    if shape_class is None:
        shape_class = Shape
    if shape_data_type is None:
        shape_data_type = np.dtype(shape_class)

    black_shapes = shapes_from_labels(
        pixels, black_labeled, black_labels, *args, shape_class=shape_class, shape_data_type=shape_data_type, **kwargs
    )

    white_shapes = shapes_from_labels(
        pixels, white_labeled, white_labels, *args, shape_class=shape_class, shape_data_type=shape_data_type, **kwargs
    )

    return black_shapes, white_shapes


def shapes_from_labels(pixels, labeled_image, labels, *args, shape_class=None, shape_data_type=None, **kwargs):
    if shape_class is None:
        shape_class = Shape
    if shape_data_type is None:
        shape_data_type = np.dtype(shape_class)

    return ndimage.labeled_comprehension(
        pixels, labeled_image, labels,
        lambda shape_pixels: shape_class(shape_pixels, *args, **kwargs),
        shape_data_type, None, False
    ).tolist()


def color_to_string(color):
    return PIXEL_VALUE_TO_COLOR_DICT.get(color)


def largest_shape_of_set(shapes):
    if shapes:
        return max(shapes, key=lambda shape: shape.area)
    return None


def smallest_shape_of_set(shapes):
    if shapes:
        return min(shapes, key=lambda shape: shape.area)
    return None


def get_connected_pixels_of_set_from_pixel(init_pixel, pixel_pool_original):
    pixel_pool = pixel_pool_original.copy()
    pixel_pool.discard(init_pixel)
    to_check = {init_pixel}
    connected_pixels = {init_pixel}
    while to_check:
        new_pixels = set()
        for pixel in to_check:
            pixels = {pixel_neighbor for pixel_neighbor in pixel if pixel_neighbor in pixel_pool}
            new_pixels.update(pixels)
            pixel_pool.difference_update(pixels)
        connected_pixels.update(new_pixels)
        to_check = new_pixels
    return connected_pixels


def get_function_name():
    return traceback.extract_stack(None, 2)[0][2]


def augmented_raise(error, message_to_add=""):
    exception_value = type(error)(str(error) + message_to_add)
    exception_traceback = sys.exc_info()[2]
    if exception_value.__traceback__ is not exception_traceback:
        raise exception_value.with_traceback(exception_traceback)
    raise exception_value
