from time import time

import numpy as np
from scipy import ndimage
import cv2

import ndarray_tools


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


class Stichographer(object):
    def __init__(self, atlas):
        self.atlas = atlas


class Morphographer(object):
    """You just add 'ographer' to everything? Now THIS is class-naming!"""

    def __init__(self, atlas, black_shapes, white_shapes, *, shape_class=None):
        self.atlas = atlas
        if shape_class is None:
            shape_class = Shape
        self._shape_class = shape_class
        self.black = black_shapes  # black_objects_array
        self.white = white_shapes  # white_objects_array
        self._black = set(black_shapes)
        self._white = set(white_shapes)
        self.primary_black = None
        self.primary_white = None
        self.current_iter = self.black

    def set_bounding_boxes_and_contained_shape_references(self):
        """Creates references of which shapes surround or own (are the smallest surrounding) other shapes."""
        self.set_all_bounding_boxes()
        self.set_all_surrounding_shapes()

    def set_all_bounding_boxes(self):
        for shape in self.black + self.white:
            shape.set_bounding_box()
            shape.set_area()
            shape.set_center()

    def set_all_surrounding_shapes(self):
        for shape in self.black + self.white:
            shape.set_surrounding_white_shapes()
            shape.set_surrounding_black_shapes()

    def __contains__(self, item):
        if isinstance(item, type(self._shape_class)):
            if item.color == 0:
                return item in self._black
            else:
                return item in self._white
        raise ValueError("Cannot check if {} in {}: {} is not a mapped type.".format(item, self.__class__, type(item)))

    def __len__(self):
        return len(self.current_iter)

    def __length_hint__(self):
        return self.current_iter.length_hint()

    def __getitem__(self, key):
        try:
            return self.current_iter[key]
        except (TypeError, IndexError):
            next_iter = self.white if self.current_iter is self.black else self.black
            try:
                return next_iter[key]
            except (TypeError, IndexError):
                raise

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        current_index = self._n
        if current_index < len(self.current_iter):
            self._n += 1
            return self.current_iter[current_index]
        raise StopIteration

    def __str__(self):
        return "<Shapeographer Object {} | {} total shapes | {} black shapes and {} white shapes.>".format(
            id(self), len(self.black) + len(self.white), len(self.black), len(self.white)
        )

    __repr__ = __str__  # Todo: repr?


class Micrographer(object):
    def __init__(self, atlas, pixels, *, pixel_class=None):
        self.atlas = atlas
        if pixel_class is None:
            pixel_class = Pixel
        self._pixel_class = pixel_class
        self.pixels = pixels  # pixel_array

    def __contains__(self, item):
        return item in self.pixels

    def __len__(self):
        return len(self.pixels)

    def __length_hint__(self):
        return self.pixels.length_hint()

    def __getitem__(self, key):
        try:
            return self.pixels[key]
        except (TypeError, IndexError):
            raise

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        current_index = self._n
        if current_index < len(self.pixels):
            self._n += 1
            return self.pixels[current_index]
        raise StopIteration

    def __str__(self):
        return "<Pixeographer Object {} | {} total pixels | {} pixels tall by {} pixels wide.>".format(
            id(self), len(self) * len(self[0]), len(self), len(self[0])
        )

    __repr__ = __str__  # Todo: repr?


class Atlas(object):
    """Coordinates references between the abstract objects in an image."""
    default_mappings = {"shapes", "pixels"}

    def __init__(self, seed_image, *, pixel_class=None, shape_class=None):
        internal_time = time()
        self.rows = len(seed_image)  # Is the original height
        self.columns = len(seed_image[0])  # Is the original width
        self.image = seed_image
        black_shapes, white_shapes, pixels = self.get_binary_four_way_connected_shapes(
            seed_image, pixel_class=pixel_class, shape_class=shape_class)
        print("Internal mapping-only completed in {}.".format(time() - internal_time))
        self.pixels = Micrographer(self, pixels, pixel_class=pixel_class)
        self.shapes = Morphographer(self, black_shapes, white_shapes, shape_class=shape_class)
        self._appendix = [self.shapes, self.pixels]

    def get_binary_four_way_connected_shapes(self, source, *, pixel_class=None, shape_class=None):
        """
        Split an image into separate black and white ConnectedPixel objects representing continuous groups of
            same-color Pixel objects.

        :Parameters:
            :param np.array source: The image to be segmented into unique binary regions.
            :param pixel_class: The instance class of objects created from image pixels.
            :param shape_class: The instance class of objects created from connected shapes.
        :rtype: Shapes list, Shapes list
        :return black_objects, white_objects, pixels: Black Shapes list, White Shapes list, Pixels list
        """
        # Convert to binary
        image = ndarray_tools.threshold_image(source, max_value=1)

        # Shapes labels
        black_labeled, black_labels, white_labeled, white_labels = binary_label_ndarray(image)

        # Init Pixels; then stack and assign pixel neighbors.
        np_pixels = quad_neighbor_pixels_from_ndarray(image, pixel_class=pixel_class)

        # Extract objects/pixels to lists
        black_shapes, white_shapes = binary_shapes_from_labels(
            np_pixels, black_labeled, black_labels, white_labeled, white_labels, self, shape_class=shape_class
        )
        pixels = np_pixels.tolist()

        return black_shapes, white_shapes, pixels

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif "_appendix" in self.__dict__:
            for internal_map in self.__dict__["_appendix"]:
                if hasattr(internal_map, name):
                    return getattr(internal_map, name)
            else:
                raise AttributeError(self._not_in(name, "getting"))
        else:
            raise AttributeError(self._not_in(name, "getting"))

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        elif hasattr(self, "_appendix"):
            for internal_map in self._appendix:
                if hasattr(internal_map, name):
                    setattr(internal_map, name, value)
                    break
            else:
                raise AttributeError(self._not_in(name, "setting"))
        else:
            self.__dict__[name] = value

    def __delattr__(self, name):
        if hasattr(self, name):
            assert name not in self.default_mappings, (
                "Error: Deleting built-in type {} will cause AttributeErrors.".format(name)
            )
            del self.__dict__[name]
        else:
            for internal_map in self._appendix:
                if hasattr(internal_map, name):
                    delattr(internal_map, name)
                    break
            else:
                raise AttributeError(self._not_in(name, "deletion"))

    def __str__(self):
        return "<Atlas Object {} | {} pixels | {} black shapes | {} white shapes | {} height x {} width.>".format(
            id(self), len(self.pixels) * len(self.pixels[0]), len(self.shapes.black), len(self.shapes.white),
            self.rows, self.columns
        )

    __repr__ = __str__  # Todo: repr?

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
        self.coordinates = coordinates  # xy_loc
        self.neighbors = (None, None, None, None)
        self.shape = None  # object_group
        self.navigation_pixel = None  # nav_pixel_ref
        # Removed: needs_checking

    def set_neighbors(self, *neighbors):
        self.neighbors = neighbors

    @staticmethod
    def black_or_white(color):
        if color == 0:
            return "Black"
        return "White"

    def __str__(self):
        return "<{} Pixel object {} from coordinates {}.>".format(
            self.black_or_white(self.color), id(self), self.coordinates
        )

    __repr__ = __str__  # Todo: repr?

    def __hash__(self):
        return hash((type(self), self.coordinates))


class Shape(object):
    """A continuous group of same-color Pixel objects representing a connected object."""

    __slots__ = ['atlas', 'assigned_row', 'assigned_column', 'assigned_segment', 'neural_network_id',
                 'is_circle', 'coordinates', 'surrounded_whitespaces',
                 'surrounded_blackspaces', 'owned_whitespaces', 'owned_blackspaces', 'smallest_surrounding_whitespace',
                 'smallest_surrounding_blackspace', 'closest_shape_for_columns', 'area', 'box', 'pixels',
                 'color', '_pixels']

    def __init__(self, init_pixels, atlas):
        """
        Set reference to owned pixel objects responsible for creation and the source space reference.

        :Parameters:
            :param Pixels ndarray or list init_pixels: The Pixels responsible for the creation of this object.
            :param Atlas atlas: The Atlas which created this object.
        :rtype: None
        :return: None
        """
        self.atlas = atlas  # space_reference
        self.assigned_row = None  # assigned_row
        self.assigned_column = None
        self.assigned_segment = None
        self.neural_network_id = None
        self.is_circle = None
        self.coordinates = None  # bounding_box_center_point
        self.surrounded_whitespaces = None
        self.surrounded_blackspaces = None
        self.owned_whitespaces = None
        self.owned_blackspaces = None
        self.smallest_surrounding_whitespace = None
        self.smallest_surrounding_blackspace = None
        self.closest_shape_for_columns = None
        self.area = None
        self.box = None  # bounding_box
        if init_pixels is not None:
            if hasattr(init_pixels, "tolist"):
                self.pixels = init_pixels.tolist()  # pixel_array
            else:
                self.pixels = list(init_pixels)
        else:
            self.pixels = []
        self.color = self.pixels[0].color  # Whitespace_or_blackspace
        self.assign_pixels(self.pixels)
        self._pixels = None

    def assign_pixels(self, pixels):
        for pix in pixels:
            pix.shape = self

    def set_pixels(self):
        self._pixels = set(self.pixels)

    def set_bounding_box(self):
        pass

    def set_area(self):
        pass

    def set_center(self):
        pass

    def set_surrounding_shapes(self, white_shapes, black_shapes):
        self.get_surrounding_shapes_from(black_shapes)

    def get_surrounding_shapes_from(self, shapes_pool):
        for whitespace in self.atlas.white_objects_array:
            inside_check = is_inside(self.bounding_box, whitespace.bounding_box)
            if inside_check and whitespace is not self:
                whitespace.update_container_status(self)
                if self.smallest_surrounding_whitespace is None:
                    self.smallest_surrounding_whitespace = whitespace
                elif whitespace.area < self.smallest_surrounding_whitespace.area:
                    self.smallest_surrounding_whitespace = whitespace
        if self.smallest_surrounding_whitespace is not None:
            shape = self.smallest_surrounding_whitespace
            shape.update_owned_shape(self)

    def placeholders(self, shapes):
        for shape in shapes:
            shape.set_bounding_box()
            shape.set_area()
            shape.set_center()
            shape.set_surrounding_white_shapes()
            shape.set_surrounding_black_shapes()


if __name__ == "__main__":
    test_im = "./test.png"
    import_img = cv2.imread(test_im, cv2.IMREAD_GRAYSCALE)
    time_1 = time()
    test_space = Atlas(import_img)
    print("Took {} seconds.".format(time() - time_1))
    print("{} black objects and {} white objects.".format(
        len(test_space.black), len(test_space.white)))
    print(test_space)
    print(test_space.pixels)
    print(test_space.shapes)
