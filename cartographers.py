from functools import wraps
from time import time

import numpy as np
from scipy import ndimage
import cv2

import ndarray_tools
import calc_tools
from py_tools import *


_BUILTIN_TABLE = "_appendix"
_BUILTIN_GETTERS = "_hidden_getters"
_BUILTIN_SETTERS = "_hidden_setters"
MINIMUM_RESEGMENT_PIXELS = 20
PIXEL_VALUE_TO_COLOR_DICT = {0: "Black", 1: "White"}


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


def resizeable(wrapped_method):
    @wraps(wrapped_method)
    def wrapper(self, *args, **kwargs):
        old_box = self.box
        old_pixels = self.pixels.copy()

        result = wrapped_method(self, *args, **kwargs)

        if len(self.pixels) != len(old_pixels):
            self.assign_pixels(self.pixels.difference(old_pixels))

            if should_recalculate:
                del self.pixel_stats
                if self.box != old_box:
                    self >> self.atlas
        return result
    return wrapper


class Stichographer(object):  # TODO
    def __init__(self, atlas):
        self.atlas = atlas


class Morphographer(object):
    """You just add 'ographer' to everything? Now THIS is class-naming!"""

    def __init__(self, atlas, *color_separated_shapes, shape_class=None, **__):
        self.atlas = atlas
        if shape_class is None:
            shape_class = Shape
        self.shape_class = shape_class
        self.shapes = {same_color_shapes[0].color: set(same_color_shapes)
                       for same_color_shapes in color_separated_shapes}
        self.primary_shape = {same_color_shapes[0].color: None for same_color_shapes
                              in color_separated_shapes}
        self.minimum_resegment_pixels = MINIMUM_RESEGMENT_PIXELS
        self.set_all_surrounding_shapes()
        self.set_primary_shapes()

    def row_map(self, *_, **__):  # TODO
        row_map = Stichographer(self.atlas)
        return row_map

    @property
    def all_shapes(self):
        all_shapes_set = set()
        for same_color_shapes in self:
            all_shapes_set.update(same_color_shapes)
        return all_shapes_set

    def set_all_surrounding_shapes(self):
        """Creates references of which shapes surround or own (are the smallest surrounding) other shapes."""
        for shape_color in self:
            for shape in shape_color:
                shape.set_surrounding_shapes(*self)

    def items(self):
        return self.shapes.items()

    def set_primary_shapes(self):
        self.primary_shape.update({shape_color_key: largest_shape_of_set(shapes_of_color)
                                   for shape_color_key, shapes_of_color in self.items()})

    def same_color_shapes_above_shape(self, shape):
        return {other_shape for other_shape in self[shape] if other_shape[0] < shape[0]}

    def same_color_shapes_below_shape(self, shape):
        return {other_shape for other_shape in self[shape] if other_shape[0] > shape[0]}

    def same_color_shapes_right_of_shape(self, shape):
        return {other_shape for other_shape in self[shape] if other_shape[1] > shape[1]}

    def same_color_shapes_left_of_shape(self, shape):
        return {other_shape for other_shape in self[shape] if other_shape[1] < shape[1]}

    def divide_shape(self, shape, source_pixels=None, shape_validator=None):
        if source_pixels is None:
            source_pixels = shape.pixels
        original_pixels = source_pixels.copy()
        check_pixels = source_pixels.copy()
        new_shapes = set()
        for pixel in original_pixels:
            if pixel in check_pixels:
                connected = get_connected_pixels_of_set_from_pixel(pixel, check_pixels)
                check_pixels.difference_update(connected)
                new_shape = Shape(connected, self.atlas)
                print("Number of pixels in new shape: {}".format(len(connected)))
                if (shape_validator(new_shape) if shape_validator is not None
                        else len(new_shape.pixels) > self.minimum_resegment_pixels):
                    new_shapes.update({new_shape})
                    shape - new_shape
                    self + new_shape
                else:
                    shape + new_shape
        return new_shapes

    def refresh_surrounding_shapes_for(self, refresh_shape):
        try:
            if any(((refresh_shape.inner or shape_root is not None) for shape_root in refresh_shape.roots.values())):
                self % refresh_shape
        except (AttributeError, TypeError):
            raise
        else:
            try:
                refresh_shape @ self
            except (TypeError, AttributeError):
                raise

    def reset_surrounding_shapes_for(self, reset_shape):
        for color_set in self:
            for shape in color_set:
                try:
                    shape % reset_shape
                except (AttributeError, TypeError):
                    raise

    def __call__(self, *args, **kwargs):
        return self.row_map(*args, **kwargs)

    def __sub__(self, other):
        try:
            self[other].discard(other)
            self % other
        except (TypeError, IndexError, KeyError, AttributeError):
            if set_list_tup(other):
                for item in other:
                    try:
                        self - item
                    except (TypeError, IndexError, KeyError, AttributeError):
                        raise
            elif isinstance(other, dict):
                for items in other.values():
                    try:
                        self - items
                    except (TypeError, IndexError, KeyError, AttributeError):
                        raise
            elif isinstance(other, type(self)):
                for shapes_of_color in other:
                    try:
                        self - shapes_of_color
                    except (TypeError, IndexError, KeyError, AttributeError):
                        raise
            else:
                raise
        return self

    def __add__(self, other):
        try:
            self[other].add(other)
            other @ self
        except (TypeError, IndexError, KeyError, AttributeError):
            if set_list_tup(other):
                for item in other:
                    try:
                        self + item
                    except (TypeError, IndexError, KeyError, AttributeError):
                        raise
            elif isinstance(other, dict):
                for items in other.values():
                    try:
                        self + items
                    except (TypeError, IndexError, KeyError, AttributeError):
                        raise
            elif isinstance(other, type(self)):
                for shapes_of_color in other:
                    try:
                        self + shapes_of_color
                    except (TypeError, IndexError, KeyError, AttributeError):
                        raise
            else:
                raise
        return self

    def __matmul__(self, other):
        try:
            self.refresh_surrounding_shapes_for(other)
        except (AttributeError, TypeError):
            if set_list_tup(other):
                for item in other:
                    try:
                        self @ item
                    except (TypeError, AttributeError):
                        raise
            elif isinstance(other, dict):
                for items in other.values():
                    try:
                        self @ items
                    except (TypeError, AttributeError):
                        raise
            elif isinstance(other, type(self)):
                for shapes_of_color in other:
                    try:
                        self @ shapes_of_color
                    except (TypeError, AttributeError):
                        raise
            else:
                raise
        return self

    def __rmatmul__(self, other):
        if set_list_tup(other):
            for item in other:
                try:
                    item @ self
                except (TypeError, AttributeError):
                    raise
        elif isinstance(other, dict):
            for items in other.values():
                try:
                    items @ self
                except (TypeError, AttributeError):
                    raise
        elif isinstance(other, type(self)):
            for shapes_of_color in other:
                try:
                    shapes_of_color @ self
                except (TypeError, AttributeError):
                    raise
        else:
            raise TypeError("Unknown instance {} being mapped by {}.".format(other, self))
        return other

    def __mod__(self, other):
        try:
            self.reset_surrounding_shapes_for(other)
        except (AttributeError, TypeError):
            if set_list_tup(other):
                for item in other:
                    try:
                        self % item
                    except (TypeError, AttributeError):
                        raise
            elif isinstance(other, dict):
                for items in other.values():
                    try:
                        self % items
                    except (TypeError, AttributeError):
                        raise
            elif isinstance(other, type(self)):
                for shapes_of_color in other:
                    try:
                        self % shapes_of_color
                    except (TypeError, AttributeError):
                        raise
            else:
                raise
        return self

    def __truediv__(self, other):
        try:
            self.divide_shape(other)
        except (AttributeError, TypeError):
            if set_list_tup(other):
                for item in other:
                    try:
                        self / item
                    except (TypeError, AttributeError):
                        raise
            elif isinstance(other, dict):
                for items in other.values():
                    try:
                        self / items
                    except (TypeError, AttributeError):
                        raise
            elif isinstance(other, type(self)):
                for shapes_of_color in other:
                    try:
                        self / shapes_of_color
                    except (TypeError, AttributeError):
                        raise
            else:
                raise
        return self

    def __getitem__(self, key):
        try:
            return self.shapes[key]
        except (TypeError, IndexError, KeyError):
            if isinstance(key, self.shape_class):
                try:
                    return self.shapes[key.color]
                except (TypeError, IndexError, KeyError, AttributeError):
                    raise
            elif set_list_tup(key) and key:
                item = next(iter(key))
                try:
                    return self[item]
                except (TypeError, IndexError, KeyError, AttributeError):
                    raise
            elif isinstance(key, dict) and key:
                try:
                    return {shape_color_key: shapes_of_same_color
                            for shape_color_key, shapes_of_same_color in self.shapes.items() if shape_color_key in key}
                except (TypeError, IndexError, KeyError, AttributeError):
                    raise
            raise

    def __contains__(self, item):
        if isinstance(item, type(self.shape_class)):
            return item in self[item.color]
        elif set_list_tup(item):
            if not item:
                raise ValueError("Cannot check if an empty {} is in {}.".format(type(item), self.__class__))
            try:
                return all((value in self for value in item))
            except (TypeError, IndexError, KeyError, AttributeError):
                raise
        elif isinstance(item, dict):
            if not item:
                raise ValueError("Cannot check if an empty {} is in {}.".format(type(item), self.__class__))
            try:
                return all((all((shape in self[shape_color_key] for shape in shapes_of_same_color))
                            for shape_color_key, shapes_of_same_color in item.items()))
            except (TypeError, IndexError, KeyError, AttributeError):
                raise
        raise TypeError("Cannot check if {} in {}: {} is not a mapped type.".format(item, self.__class__, type(item)))

    def __bool__(self):
        return any((shapes for shapes in self))

    def __delitem__(self, key):
        try:
            self[key].discard(key)
        except (TypeError, IndexError, KeyError, AttributeError):
            try:
                self[key].difference_update(key)
            except (TypeError, IndexError, KeyError, AttributeError):
                try:
                    self - key
                except (TypeError, IndexError, KeyError, AttributeError):
                    raise
        return self

    def __len__(self):
        return len(self.shapes.values())

    def __iter__(self):
        return iter(self.shapes.values())

    def __str__(self):
        return ("<Shapeographer Object {} | {} total shapes | ".format(id(self), len(self.all_shapes))
                + ' | '.join(("{} {} shapes".format(len(shapes_of_color), color_to_string(color_key))
                              for color_key, shapes_of_color in self.items()))
                + ">")

    __repr__ = __str__  # Todo: repr?


class Pixeographer(object):
    def __init__(self, atlas):
        self.atlas = atlas
        self.pixel_class = None
        self.pixels = []

    def four_connected_binary_map(self, image, *_,
                                  image_threshold=ndarray_tools.DEFAULT_IMAGE_THRESH,
                                  pixel_class=None,
                                  shape_class=None):

        assert not self.pixels, "Pixeographer already mapping pixels, cannot map another image."
        if pixel_class is None:
            pixel_class = Pixel
        self.pixel_class = pixel_class

        # Convert to binary
        image = ndarray_tools.threshold_image(image, threshold=image_threshold, max_value=1)

        # Shapes labels
        black_labeled, black_labels, white_labeled, white_labels = binary_label_ndarray(image)

        # Init Pixels; then stack and assign pixel neighbors.
        np_pixels = quad_neighbor_pixels_from_ndarray(image, pixel_class=pixel_class)

        # Extract objects/pixels to lists
        shape_map = Morphographer(
            self.atlas, *binary_shapes_from_labels(
                np_pixels, black_labeled, black_labels, white_labeled, white_labels, self.atlas, shape_class=shape_class
            ), shape_class=shape_class
        )
        self.pixels = np_pixels.tolist()
        return shape_map

    def __call__(self, *args, **kwargs):
        return self.four_connected_binary_map(*args, **kwargs)

    def __contains__(self, item):
        return item in self.pixels

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, key):
        try:
            return self.pixels[key]
        except (TypeError, IndexError, KeyError):
            if isinstance(key, type(self.pixel_class)):
                return self.pixel_class
            elif set_list_tup(key) and key:
                item = next(iter(key))
                try:
                    return self[item]
                except (TypeError, IndexError, KeyError):
                    raise
            raise

    def __delitem__(self, key):
        try:
            del self.pixels[key]
        except (TypeError, IndexError, KeyError):
            try:
                for item in key:
                    del self.pixels[item.coordinates]
            except (TypeError, IndexError, KeyError, ValueError):
                raise

    def __iter__(self):
        self._index_0 = 0
        self._index_1 = 0
        return self

    def __next__(self):
        if self._index_0 < len(self.pixels):
            dimension_0_current = self._index_0
            self._index_0 += 1
            return self.pixels[dimension_0_current][self._index_1]
        elif self._index_1 < len(self.pixels[0]) - 1:
            self._index_0 = 0
            self._index_1 += 1
            return self.pixels[self._index_0][self._index_1]
        raise StopIteration

    def __str__(self):
        return "<Pixeographer Object {} | {} total pixels | {} pixels tall by {} pixels wide>".format(
            id(self), len(self) * len(self[0]), len(self), len(self[0])
        )

    __repr__ = __str__  # Todo: repr?


class Atlas(object):
    """Coordinates references between the abstract objects in an image."""
    default_mappings = {"shape_map", "pixel_map", "row_map", _BUILTIN_TABLE}

    def __init__(self, seed_image, *mapping_args, **mapping_kwargs):
        internal_time = time()
        self.rows = len(seed_image)
        self.columns = len(seed_image[0])
        self.image = seed_image
        self.pixel_map = Pixeographer(self)
        self.shape_map = self.pixel_map(seed_image, *mapping_args, **mapping_kwargs)
        self.row_map = self.shape_map(*mapping_args, **mapping_kwargs)
        self._hidden_getters = {"page": lambda: self._appendix[0]}
        self._hidden_setters = {"page": lambda page: self._turn_page(self._appendix.index(page))}
        self._appendix = [self.shape_map, self.pixel_map, self.row_map]
        self.page = self.shape_map
        fin_time = time()
        print("Total Atlas Time: {}.".format(fin_time - internal_time))

    def get_cartographer(self, item):
        try:
            cartographer, _ = self._get_cartographer_and_value(item)
            return cartographer
        except KeyError:
            raise

    def _get_cartographer_and_value(self, item):
        if hasattr(self, _BUILTIN_TABLE):
            for internal_map in self.__dict__[_BUILTIN_TABLE]:
                try:
                    return internal_map, internal_map[item]
                except (TypeError, IndexError, KeyError, AttributeError):
                    pass
        if item in self.__dict__:
            return self, getattr(self, item)
        raise KeyError("Key {} not in {}.".format(item, self))

    def _turn_page(self, key=-1):
        if key < 0:
            self._appendix = self._appendix[key:key]
        elif key > 0:
            self._appendix = self._appendix[key:] + self._appendix[:key]

    def __iter__(self):
        return iter(self.page)

    def __getitem__(self, key):
        if _BUILTIN_TABLE in self.__dict__:
            for internal_map in self.__dict__[_BUILTIN_TABLE]:
                try:
                    return internal_map[key]
                except (TypeError, IndexError, KeyError, AttributeError):
                    pass
        raise KeyError("Key {} not in {}.".format(key, self))

    def __getattr__(self, name):
        if _BUILTIN_TABLE in self.__dict__:
            for internal_map in self.__dict__[_BUILTIN_TABLE]:
                if hasattr(internal_map, name):
                    return getattr(internal_map, name)
            if name in self.__dict__:
                return self.__dict__[name]
            elif name in self.__dict__[_BUILTIN_GETTERS]:
                return self.__dict__[_BUILTIN_GETTERS].get(name)()
            raise AttributeError(self._not_in(name, "getting"))
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError(self._not_in(name, "getting"))

    def __setattr__(self, name, value):
        if hasattr(self, _BUILTIN_TABLE):
            for internal_map in self.__dict__[_BUILTIN_TABLE]:
                if hasattr(internal_map, name):
                    setattr(internal_map, name, value)
                    break
            else:
                if name in self.__dict__:
                    self.__dict__[name] = value
                elif name in self.__dict__[_BUILTIN_SETTERS]:
                    self.__dict__[_BUILTIN_SETTERS].get(name)(value)
                else:
                    raise AttributeError(self._not_in(name, "setting"))
        else:
            self.__dict__[name] = value

    def __delattr__(self, name):
        if hasattr(self, name):
            if name in self.default_mappings:
                raise ValueError("Deleting built-in type {} will cause AttributeErrors.".format(name))
            del self.__dict__[name]
        else:
            for internal_map in self._appendix:
                if hasattr(internal_map, name):
                    delattr(internal_map, name)
                    break
            else:
                raise AttributeError(self._not_in(name, "deletion"))

    def __str__(self):
        return ("<Atlas Object {} | {} pixels | ".format(
                    id(self), len(self.pixels) * len(self.pixels[0]))
                + ' | '.join(("{} {} shapes".format(
                    len(shapes_of_color), color_to_string(color_key))
                    for color_key, shapes_of_color in self.shapes.items()))
                + " | {} height x {} width>".format(
                    self.rows, self.columns))

    __repr__ = __str__  # Todo: repr?

    def redirect_operator(self, operator, operand):
        try:
            cartographer = self.get_cartographer(operand)
            print("CART: {}".format(cartographer))
        except KeyError:
            raise
        else:
            return getattr(cartographer, operator)(operand)

    def __sub__(self, other):
        return self.redirect_operator(get_function_name(), other)

    def __add__(self, other):
        return self.redirect_operator(get_function_name(), other)

    def __mod__(self, other):
        return self.redirect_operator(get_function_name(), other)

    def __matmul__(self, other):
        return self.redirect_operator(get_function_name(), other)

    def __rmatmul__(self, other):
        return self.redirect_operator(get_function_name(), other)

    def __truediv__(self, other):
        return self.redirect_operator(get_function_name(), other)

    def __delitem__(self, other):
        return self.redirect_operator(get_function_name(), other)

    @classmethod
    def _not_in(cls, name, access_method):
        return "Attribute {} not mapped in {} for {}.".format(name, cls.__name__, access_method)


class Pixel(object):
    """Object representing individual pixel values and indices in an image."""

    __slots__ = ['color', 'coordinates', 'shape', 'neighbors', 'navigation_pixel']

    def __init__(self, color, coordinates):
        """
        "Set color and indices from source image."

        :Parameters:
            :param int color: The color of the thresholded source image pixel, should be either 1 or 0.
            :param ints tuple coordinates: The (column, row) index of the source image pixel.
        :rtype: None
        :return: None
        """
        self.color = color
        self.coordinates = coordinates
        self.neighbors = (None, None, None, None)
        self.shape = None
        self.navigation_pixel = None

    def set_neighbors(self, *neighbors):
        self.neighbors = neighbors

    def __sub__(self, other):
        return tuple(coord - other_coord for coord, other_coord in zip(self.coordinates, other.coordinates))

    def __lt__(self, other):
        return tuple(coord < other_coord for coord, other_coord in zip(self.coordinates, other.coordinates))

    def __le__(self, other):
        return tuple(coord <= other_coord for coord, other_coord in zip(self.coordinates, other.coordinates))

    def __gt__(self, other):
        return tuple(coord > other_coord for coord, other_coord in zip(self.coordinates, other.coordinates))

    def __ge__(self, other):
        return tuple(coord >= other_coord for coord, other_coord in zip(self.coordinates, other.coordinates))

    def __eq__(self, other):
        return tuple(coord == other_coord for coord, other_coord in zip(self.coordinates, other.coordinates))

    def __contains__(self, item):
        return item in self.neighbors

    def __len__(self):
        return len(self.neighbors)

    def __getitem__(self, key):
        try:
            return self.neighbors[key]
        except (TypeError, IndexError):
            raise

    def __iter__(self):
        return iter(self.neighbors)

    def __str__(self):
        return "<{} Pixel object {} from coordinates {}>".format(
            color_to_string(self.color), id(self), self.coordinates
        )

    __repr__ = __str__  # Todo: repr?

    def __hash__(self):
        return hash((type(self), id(self)))


class Shape(object):
    """A continuous group of same-color Pixel objects representing a connected object."""

    __slots__ = ['atlas', 'row', 'column', 'segment', 'inner', 'owned', 'roots', 'pixels', 'color',
                 '_coordinates', '_area', '_height', '_width', '_box']

    def __init__(self, init_pixels, atlas):
        """
        Set reference to owned pixel objects responsible for creation and the source space reference.

        :Parameters:
            :param Pixels ndarray or list init_pixels: The Pixels responsible for the creation of this object.
            :param Atlas atlas: The Atlas which created this object.
        :rtype: None
        :return: None
        """
        self.atlas = atlas
        self.row = None
        self.column = None
        self.segment = None
        self.inner = set()
        self.owned = set()
        self.roots = {0: None, 1: None}
        self._coordinates = None
        self._area = None
        self._height = None
        self._width = None
        self._box = None
        if init_pixels is not None:
            if hasattr(init_pixels, "tolist"):
                self.pixels = set(init_pixels.tolist())
            else:
                self.pixels = set(init_pixels)
            self.color = next(iter(self.pixels)).color if self.pixels else None
        else:
            self.pixels = set()
            self.color = None
        self.assign_pixels(self.pixels)

    def assign_pixels(self, pixels):
        for pix in pixels:
            pix.shape = self

    def set_surrounding_shapes(self, *color_separated_shapes):
        for color_set in color_separated_shapes:
            self.set_surrounding_shapes_from_single_color_set(color_set)

    def set_surrounding_shapes_from_single_color_set(self, shapes_pool):
        smallest_shape = None
        smallest_area = None
        for shape in shapes_pool:
            _, smallest_shape, smallest_area = self.compare_shape_context(shape, smallest_shape, smallest_area)
        self.set_root(smallest_shape)

    def compare_shape_context(self, shape, previous_smallest_shape, previous_smallest_area):
        inside_check = calc_tools.is_inside(*self.box, *shape.box)
        if inside_check and shape is not self:
            shape.insert(self)
            try:
                if shape.area < previous_smallest_area:
                    previous_smallest_shape = shape
                    previous_smallest_area = shape.area
            except TypeError:
                previous_smallest_shape = shape
                previous_smallest_area = shape.area
        return inside_check, previous_smallest_shape, previous_smallest_area

    def set_root(self, smallest_shape):
        if smallest_shape is not None:
            old_root = self.roots.get(smallest_shape.color)
            if old_root is not None:
                old_root.pop(self)
            smallest_shape.insert(self, 1)
            self.roots.update({smallest_shape.color: smallest_shape})

    def set_shapes_context(self, *color_separated_shapes):
        for color_set in color_separated_shapes:
            self.set_shape_context_from_single_color_set(color_set)

    def set_shape_context_from_single_color_set(self, shapes_pool):
        smallest_shape = None
        smallest_area = None
        for shape in shapes_pool:
            if shape is not self:
                inside, smallest_shape, smallest_area = self.compare_shape_context(shape, smallest_shape, smallest_area)
                if not inside:
                    inside_check = calc_tools.is_inside(*shape.box, *self.box)
                    if inside_check:
                        shape.compare_root(self)
        self.set_root(smallest_shape)

    def compare_root(self, new_root):
        old_root = self.roots.get(new_root.color)
        if old_root is not None:
            if old_root.area > new_root.area:
                self.set_root(new_root)
            else:
                new_root.insert(self)
        else:
            self.set_root(new_root)

    def get_neighbor_pixels_of_shape(self, shape):
        neighbor_pixels = {}
        for pixel in self.pixels:
            for neighbor in pixel:
                if neighbor and neighbor.shape is shape:
                    neighbor_pixels.update({neighbor})
        return neighbor_pixels

    def copy_of_pixels_and(self, other):
        if other:
            item = next(iter(other))
            if isinstance(item, type(self.atlas.pixel_class)):
                return self.pixels.union(other)
            elif isinstance(item, type(self)):
                self_copy = self.pixels.copy()
                for shape in other:
                    self_copy.update(shape.pixels)
                return self_copy
            raise TypeError(
                "Unexpected container with items of type {} finding sum pixels with {}.".format(
                    type(item), self
                )
            )
        return self.pixels.copy()

    def copy_of_pixels_except(self, other):
        if other:
            item = next(iter(other))
            if isinstance(item, type(self.atlas.pixel_class)):
                return self.pixels.difference(other)
            elif isinstance(item, type(self)):
                self_copy = self.pixels.copy()
                for shape in other:
                    self_copy.difference_update(shape.pixels)
                return self_copy
            raise TypeError(
                "Unexpected container with items of type {} finding pixel difference with {}.".format(
                    type(item), self
                )
            )
        return self.pixels.copy()

    def take_pixels(self, pixels_source):
        if isinstance(pixels_source, type(self)):
            self.pixels.update(pixels_source.pixels)
        elif set_list_tup(pixels_source):
            try:
                self.extend(pixels_source)
            except TypeError:
                raise
        elif isinstance(pixels_source, self.atlas.pixel_class):
            self.pixels.update({pixels_source})
        else:
            raise TypeError(
                "Unexpected item of type {} adding pixels to {}.".format(
                    type(pixels_source), self
                )
            )
        return self

    def extend(self, other):
        if other:
            item = next(iter(other))
            if isinstance(item, type(self.atlas.pixel_class)):
                self.pixels.update(other)
            elif isinstance(item, type(self)):
                for shape in other:
                    self.pixels.update(shape.pixels)
            else:
                raise TypeError(
                    "Unexpected container with items of type {} adding pixels to {}.".format(
                        type(item), self
                    )
                )

    def remove_pixels(self, pixels_source):
        if isinstance(pixels_source, type(self)):
            self.pixels.difference_update(pixels_source.pixels)
        elif set_list_tup(pixels_source):
            try:
                self.difference(pixels_source)
            except TypeError:
                raise
        elif isinstance(pixels_source, self.atlas.pixel_class):
            self.pixels.difference_update({pixels_source})
        else:
            raise TypeError(
                "Unexpected item of type {} removing pixels from {}.".format(
                    type(pixels_source), self
                )
            )
        return self

    def difference(self, other):
        if other:
            item = next(iter(other))
            if isinstance(item, type(self.atlas.pixel_class)):
                self.pixels.difference_update(other)
            elif isinstance(item, type(self)):
                for shape in other:
                    self.pixels.difference_update(shape.pixels)
            else:
                raise TypeError(
                    "Unexpected container with items of type {} removing pixels from {}.".format(
                        type(item), self
                    )
                )

    def insert(self, other, depth=2):
        assert other is not None, "Cannot insert None into shape context."
        try:
            if depth == 1:
                self.owned.add(other)
            self.inner.add(other)
        except TypeError:
            for item in other:
                try:
                    self.insert(item, depth)
                except TypeError as e:
                    augmented_raise(e, "Unexpected instance {} inserted in {}.".format(other, self))
        return self

    def pop(self, other=None, depth=1):
        assert other is not None, "Cannot remove None from shape context."
        try:
            if depth == 1:
                self.owned.discard(other)
            else:
                self.owned.discard(other)
                self.inner.add(other)
        except TypeError:
            for item in other:
                try:
                    self.pop(item, depth)
                except TypeError as e:
                    augmented_raise(e, "Unexpected instance {} pop'd in {}.".format(other, self))
        return self

    def index(self, item):
        try:
            if item in self:
                return 1
        except TypeError as e:
            augmented_raise(e, "Unhashable instance {} queried as contained in {}.".format(item, self))
        else:
            if item in self.inner:
                return 2
            elif item in self.pixels:
                return True
            return False

    def recalculate_linked_contexts(self, context_object=None):
        if context_object is None:
            context_object = self.atlas
        original_owned = self.owned.copy()
        context_object % self
        tuple((shape @ context_object for shape in original_owned))
        print(len(self.pixels))
        if self:
            self @ context_object
            return self
        del context_object[self]

    def copy(self):
        return self.pixels.copy()

    @property
    def box(self):
        if self._box is None:
            self.box = self.pixels
        return self._box

    @box.setter
    def box(self, pixels):
        if pixels is not None:
            if pixels:
                min_vertical = min(pixel.coordinates[0] for pixel in pixels)
                max_vertical = max(pixel.coordinates[0] for pixel in pixels)
                min_horizontal = min(pixel.coordinates[1] for pixel in pixels)
                max_horizontal = max(pixel.coordinates[1] for pixel in pixels)
            else:
                max_vertical = min_vertical = max_horizontal = min_horizontal = -1
            self._box = (min_vertical, max_vertical, min_horizontal, max_horizontal)
        else:
            self._box = None

    @property
    def height(self):
        if self._height is None:
            self.dimensions = self.box
        return self._height

    @height.deleter
    def height(self):
        self._height = None

    @property
    def width(self):
        if self._width is None:
            self.dimensions = self.box
        return self._width

    @width.deleter
    def width(self):
        self._width = None

    @property
    def area(self):
        if self._area is None:
            self._area = self.height * self.width
        return self._area

    @area.deleter
    def area(self):
        self._area = None

    @property
    def dimensions(self):
        if None in (self._height, self._width):
            self.dimensions = self.box
        return self.height, self.width

    @dimensions.setter
    def dimensions(self, box):
        self._height, self._width = calc_tools.find_dimension_from_box(*box)

    @dimensions.deleter
    def dimensions(self):
        self._height, self._width = None, None
        self._area = None

    @property
    def coordinates(self):
        if self._coordinates is None:
            self._coordinates = calc_tools.find_coordinates_from_box(*self.box)
        return self._coordinates

    @coordinates.deleter
    def coordinates(self):
        self._coordinates = None

    @property
    def pixel_stats(self):
        return self.box, self.area, self.coordinates

    @pixel_stats.deleter
    def pixel_stats(self):
        self._box = None
        self._area = None
        self._coordinates = None

    def __getitem__(self, key):
        try:
            return self.coordinates[key]
        except (TypeError, IndexError):
            raise

    def __contains__(self, item):
        if isinstance(item, type(self)):
            return item in self.owned
        elif isinstance(item, type(self.atlas.pixel_class)):
            return item in self.pixels
        raise TypeError(
            "Unexpected item of type {} queried as contained in {}.".format(
                type(item), self
            )
        )

    def __bool__(self):
        return bool(self.pixels)

    def __len__(self):
        return len(self.owned)

    def __iter__(self):
        return iter(self.owned)

    def __mod__(self, other):
        return self.pop(other, 2)

    def __matmul__(self, shape_sets_per_color):
        return self.set_shapes_context(*shape_sets_per_color)

    def __rshift__(self, other):
        return self.recalculate_linked_contexts(other)

    def __rlshift__(self, other):
        return self >> other

    def __neg__(self):
        del self.atlas[self]

    def __delitem__(self, key):
        return self - key

    @resizeable
    def __add__(self, other):
        self.take_pixels(other)
        return self

    @resizeable
    def __sub__(self, other):
        self.remove_pixels(other)
        return self

    def __lt__(self, other):
        return other.index(self)

    def __le__(self, other):
        return self in other

    def __eq__(self, other):
        return self.pixels == other.pixels

    def __gt__(self, other):
        return self.index(other)

    def __ge__(self, other):
        return other in self

    def __str__(self):
        return "<{} Shape Object {} | {} total pixels | {} area | {} height x {} width>".format(
            color_to_string(self.color), id(self), len(self.pixels), self.area, self.height, self.width
        )

    __repr__ = __str__

    def __hash__(self):
        return hash((type(self), id(self)))


def test_resegment(atlas):
    count = 0
    previous_shapes = len(atlas[0])
    for black in atlas[0].copy():
        previous_pixels = len(black.pixels)
        new_blacks = atlas.divide_shape(black)
        new_pixels = len(black.pixels)
        print("Previous # Pixels: {}\nRemaining # Pixels: {}".format(previous_pixels,
                                                                     new_pixels))
        print("New Shapes:")
        for new_black in new_blacks:
            print(new_black)
        count += 1
        if count > 5:
            break
    new_shapes = len(atlas[0])
    print("New # shapes: {}".format(new_shapes - previous_shapes))


if __name__ == "__main__":
    test_im = "./test.png"
    import_img = cv2.imread(test_im, cv2.IMREAD_GRAYSCALE)
    time_1 = time()
    test_space = Atlas(import_img)
    print("Took {} seconds.".format(time() - time_1))
    print("{} black objects and {} white objects.".format(
        len(test_space[0]), len(test_space[1])))
    print(test_space)
    test_resegment(test_space)
    # print(test_space.pixels)
    # print(test_space.shapes)
