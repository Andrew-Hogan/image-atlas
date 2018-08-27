import functools
import calc_tools
import atlas_tools as at


def resizeable(_wrapped_method, *, _=None):

    def resizeable_decorator(wrapped_method):
        @functools.wraps(wrapped_method)
        def wrapper(self, *args, **kwargs):
            old_box = self.box
            old_pixels = self.pixels.copy()

            result = wrapped_method(self, *args, **kwargs)

            if len(self.pixels) != len(old_pixels):
                self.assign_pixels(self.pixels.difference(old_pixels))

                if at.should_recalculate:
                    del self.pixel_stats
                    if self.box != old_box:
                        self >> self.atlas
            return result
        return wrapper

    if _wrapped_method is None:
        return resizeable_decorator
    return resizeable_decorator(_wrapped_method)


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
        try:
            return tuple(coord == other_coord for coord, other_coord in zip(self.coordinates, other.coordinates))
        except AttributeError:
            return False

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
            at.color_to_string(self.color), id(self), self.coordinates
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

    def owned_of(self, color):
        return [shape for shape in self if shape.color == color]

    def inner_of(self, color):
        return [shape for shape in self.inner if shape.color == color]

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
        elif isinstance(pixels_source, (set, list, tuple)):
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
        elif isinstance(pixels_source, (set, list, tuple)):
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
                    at.augmented_raise(e, "Unexpected instance {} inserted in {}.".format(other, self))
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
                    at.augmented_raise(e, "Unexpected instance {} pop'd in {}.".format(other, self))
        return self

    def index(self, item):
        try:
            if item in self:
                return 1
        except TypeError as e:
            at.augmented_raise(e, "Unhashable instance {} queried as contained in {}.".format(item, self))
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
    def distances_from_center(self):
        return [calc_tools.find_distance(*self.coordinates, *pixel.coordinates) for pixel in self.pixels]

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
        try:
            return self.pixels == other.pixels
        except AttributeError:
            return False

    def __gt__(self, other):
        return self.index(other)

    def __ge__(self, other):
        return other in self

    def __str__(self):
        return "<{} Shape Object {} | {} total pixels | {} area | {} height x {} width>".format(
            at.color_to_string(self.color), id(self), len(self.pixels), self.area, self.height, self.width
        )

    __repr__ = __str__

    def __hash__(self):
        return hash((type(self), id(self)))
