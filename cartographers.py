from time import time

import cv2

from atlas_tools import *
import inspectors


_BUILTIN_TABLE = "_appendix"
_BUILTIN_GETTERS = "_hidden_getters"
_BUILTIN_SETTERS = "_hidden_setters"
MINIMUM_RESEGMENT_PIXELS = 20


def redirectable(_wrapped_method=None, *,
                 exceptions=(TypeError, IndexError, KeyError, AttributeError),
                 in_iterable_validator=lambda x, y: isinstance(y, (set, list, tuple, x.__class__)),
                 in_items_validator=lambda x, y: isinstance(y, dict),
                 args_target_index=0,
                 iter_results_post_processor=lambda results: results[-1]):

    args_and_target_mesh = (
        lambda args_tuple, target: args_tuple[args_target_index+1:] + (target,) + args_tuple[:args_target_index]
    )

    def decorate_redirect(wrapped_method):
        @functools.wraps(wrapped_method)
        def wrapper(self, *args, **kwargs):
            try:
                return wrapped_method(self, *args, **kwargs)
            except exceptions:
                redirect_target = args[args_target_index]
                ret = []
                if in_iterable_validator(self, redirect_target):
                    try:
                        for item in redirect_target:
                            ret.append(wrapped_method(self, *args_and_target_mesh(args, item), **kwargs))
                    except exceptions:
                        raise
                    return iter_results_post_processor(ret)
                elif in_items_validator(self, redirect_target):
                    try:
                        for item in redirect_target.items():
                            ret.append(wrapped_method(self, *args_and_target_mesh(args, item), **kwargs))
                    except exceptions:
                        raise
                    return iter_results_post_processor(ret)
                raise
        return wrapper

    if _wrapped_method is None:
        return decorate_redirect
    return decorate_redirect(_wrapped_method)


def annex(_wrapped_method=None, *, swap_order=False):
    def decorate_annex(wrapped_method):
        @functools.wraps(wrapped_method)
        def wrapper(self, other, *args, **kwargs):
            try:
                relevant_mapper = wrapped_method(self, other, *args, **kwargs)
            except KeyError:
                raise
            if swap_order:
                return getattr(other, wrapped_method.__name__)(relevant_mapper, *args, **kwargs)
            return getattr(relevant_mapper, wrapped_method.__name__)(other, *args, **kwargs)
        return wrapper

    if _wrapped_method is None:
        return decorate_annex
    return decorate_annex(_wrapped_method)


class Atlas(object):
    """Coordinates references between the abstract objects in an image."""
    default_mappings = {"shape_map", "pixel_map", "row_map", _BUILTIN_TABLE}

    def __init__(self, seed_image_or_file, *mapping_args, **mapping_kwargs):
        internal_time = time()
        self._annex_key_converters = {
            list.__name__: lambda x: x[0],
            tuple.__name__: lambda x: x[0],
            set.__name__: lambda x: next(iter(x)),
            dict.__name__: lambda x: x.values()[0]
        }
        self._annex = {}
        self.image_lens = inspectors.Lens(self)
        self.pixel_map = Pixeographer(self)
        self.shape_map = self.pixel_map(self.image_lens(seed_image_or_file), *mapping_args, **mapping_kwargs)
        self.row_map = self.shape_map(*mapping_args, **mapping_kwargs)
        self._hidden_getters = {"page": lambda: self._appendix[0]}
        self._hidden_setters = {"page": lambda page: self._turn_page(self._get_appendix_index_for(page))}
        self._appendix = [self.shape_map, self.pixel_map, self.row_map]
        self.page = self.shape_map
        fin_time = time()
        print("Total Atlas Time: {}.".format(fin_time - internal_time))

    def update_annex_key_converters(self, converter_dict):
        for key, converter in converter_dict.items():
            if key in self._annex_key_converters:
                raise ValueError("Changing built-in converter for class {} will cause AttributeErrors.".format(key))
            self._annex_key_converters.update({key: converter})

    def update_annex(self, annex_to_cartographer_dict):
        self._annex.update(annex_to_cartographer_dict)

    def open_to_page(self, topic_in_page):
        try:
            self.page = self.get_cartographer(topic_in_page)
        except KeyError:
            raise
        else:
            return self

    def get_cartographer(self, item):
        annex_key_converter = self._annex_key_converters.get(item.__class__.__name__, lambda x: x)
        annex_key = annex_key_converter(item).__class__.__name__
        try:
            cartographer = self._annex.get(annex_key, self.get_cartographer_and_set(item)[0])
        except KeyError:
            raise
        else:
            if annex_key not in self._annex:
                self._annex.update({annex_key: cartographer})
            return cartographer

    def get_cartographer_and_set(self, item):
        if hasattr(self, _BUILTIN_TABLE):
            for internal_map in self.__dict__[_BUILTIN_TABLE]:
                try:
                    return internal_map, internal_map[item]
                except (TypeError, IndexError, KeyError, AttributeError):
                    pass
        if item in self.__dict__:
            return self, getattr(self, item)
        raise KeyError("Key {} not in {}.".format(item, self))

    def _get_appendix_index_for(self, page_or_item):
        if page_or_item in self._appendix:
            return self._appendix.index(page_or_item)
        else:
            new_page = self.get_cartographer(page_or_item)
            return self._appendix.index(new_page)

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
        return (
            "<Atlas Object {} | {} pixels | ".format(id(self), len(self.pixels) * len(self.pixels[0]))
            + ' | '.join(("{} {} shapes".format(len(shapes_of_color), color_to_string(color_key))
                          for color_key, shapes_of_color in self.shapes.items()))
            + " | {} height x {} width>".format(self.image_lens.rows, self.image_lens.columns)
        )

    __repr__ = __str__

    @annex
    def __sub__(self, other):
        return self.get_cartographer(other)

    @annex
    def __add__(self, other):
        return self.get_cartographer(other)

    @annex
    def __mod__(self, other):
        return self.get_cartographer(other)

    @annex
    def __matmul__(self, other):
        return self.get_cartographer(other)

    @annex(swap_order=True)
    def __rmatmul__(self, other):
        return self.get_cartographer(other)

    @annex
    def __truediv__(self, other):
        return self.get_cartographer(other)

    @annex
    def __delitem__(self, other):
        return self.get_cartographer(other)

    @classmethod
    def _not_in(cls, name, access_method):
        return "Attribute {} not mapped in {} for {}.".format(name, cls.__name__, access_method)


class Pixeographer(object):
    def __init__(self, atlas):
        self.atlas = atlas
        self.pixel_class = None
        self.pixels = []

    def four_connected_binary_map(self, image, *_,
                                  pixel_class=None,
                                  shape_class=None, **__):

        assert not self.pixels, "Pixeographer already mapping pixels, cannot map another image."
        if pixel_class is None:
            pixel_class = Pixel
        self.pixel_class = pixel_class
        self.atlas.update_annex({self.pixel_class.__name__: self})

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
            elif isinstance(key, (set, list, tuple)) and key:
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


class Morphographer(object):
    """You just add 'ographer' to everything? Now THIS is class-naming!"""

    def __init__(self, atlas, *color_separated_shapes, shape_class=None, **__):
        self.atlas = atlas
        if shape_class is None:
            shape_class = Shape
        self.shape_class = shape_class
        self.atlas.update_annex({self.shape_class.__name__: self})
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

    def keys(self):
        return self.shapes.keys()

    def values(self):
        return self.shapes.values()

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

    def _instance_dispatch(self, other, call):
        if isinstance(other, (set, list, tuple, self.__class__)):
            for item in other:
                call(self, item)
        elif isinstance(other, dict):
            for item in other.values():
                call(self, item)
        else:
            return False
        return True

    def __call__(self, *args, **kwargs):
        return self.row_map(*args, **kwargs)

    @redirectable
    def __sub__(self, other):
        try:
            print(type(other))
            self[other].discard(other)
            self % other
        except (TypeError, IndexError, KeyError, AttributeError):
            raise
        return self

    @redirectable
    def __add__(self, other):
        try:
            self[other].add(other)
            other @ self
        except (TypeError, IndexError, KeyError, AttributeError):
            raise
        return self

    @redirectable
    def __matmul__(self, other):
        try:
            self.refresh_surrounding_shapes_for(other)
        except (AttributeError, TypeError):
            raise
        return self

    @redirectable
    def __rmatmul__(self, other):
        # if not self._instance_dispatch(other, lambda x, y: y @ x):
        if hasattr(other, get_function_name()):
            other @ self
        else:
            raise TypeError("Unknown instance {} being mapped by {}.".format(other, self))
        return other

    @redirectable
    def __mod__(self, other):
        try:
            self.reset_surrounding_shapes_for(other)
        except (AttributeError, TypeError):
            raise
        return self

    @redirectable
    def __truediv__(self, other):
        try:
            self.divide_shape(other)
        except (AttributeError, TypeError):
            raise
        return self

    # iter_results_post_processor=lambda results: max(results, key=lambda x: results.count(x))
    @redirectable(
        iter_results_post_processor=lambda results: [y for x, y in enumerate(results) if results.index(y) == x]
    )
    def __getitem__(self, key):
        try:
            return self.shapes[key]
        except (TypeError, IndexError, KeyError, AttributeError):
            try:
                return self.shapes[key.color]
            except (TypeError, IndexError, KeyError, AttributeError):
                try:
                    return {shape_color_key: shapes_of_same_color
                            for shape_color_key, shapes_of_same_color in self.items() if shape_color_key in key.keys()}
                except (TypeError, IndexError, KeyError, AttributeError):
                    raise

    @redirectable(iter_results_post_processor=lambda results: all(results))
    def __contains__(self, item):
        if isinstance(item, type(self.shape_class)):
            return item in self[item.color]
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

    @redirectable
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
        return ("<Morphographer Object {} | {} total shapes | ".format(id(self), len(self.all_shapes))
                + ' | '.join(("{} {} shapes".format(len(shapes_of_color), color_to_string(color_key))
                              for color_key, shapes_of_color in self.items()))
                + ">")

    __repr__ = __str__  # Todo: repr?


class Stichographer(object):  # TODO
    def __init__(self, atlas):
        self.atlas = atlas


class RedirectTester(object):
    def __init__(self, values):
        self.values = values

    @redirectable(
        iter_results_post_processor=lambda results: [y for x, y in enumerate(results) if results.index(y) == x]
    )
    def __getitem__(self, item):
        return self.values[item]


def test_resegment(atlas):
    count = 0
    previous_shapes = len(atlas[0])
    save_new = []
    for black in atlas[0].copy():
        previous_pixels = len(black.pixels)
        new_blacks = atlas.divide_shape(black)
        new_pixels = len(black.pixels)
        print("Previous # Pixels: {}\nRemaining # Pixels: {}".format(previous_pixels,
                                                                     new_pixels))
        print("New Shapes:")
        for new_black in new_blacks:
            print(new_black)
            save_new.append(new_black)
        count += 1
        if count > 5:
            break
    test_pixel = next(iter(save_new[0].pixels))
    curr_page = len([x for x in atlas])

    print("Current page: {} | Length of current page: {}".format(atlas.page, curr_page))

    atlas.open_to_page(test_pixel)

    new_page = len([x for x in atlas])

    print("Current page: {} | Length of current page: {}".format(atlas.page, new_page))

    atlas.page = save_new[0]

    final_page = len([x for x in atlas])

    print("Current page: {} | Length of current page: {}".format(atlas.page, final_page))

    atlas - save_new
    new_shapes = len(atlas[0])
    print("New # shapes: {}".format(new_shapes - previous_shapes))
    print("Atlas annex: {}".format(atlas._annex))
    return


def redirectable_test():
    test_inner_values = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
    test_object = RedirectTester(test_inner_values)
    assert test_object[0] == test_inner_values[0]
    assert test_object[2] == test_inner_values[2]
    outer_tests = [[3, 1, 1, 1]]
    print(outer_tests[0].index(1))
    for test_outer_values in outer_tests:
        print(test_object[test_outer_values])
    sys.exit(0)


if __name__ == "__main__":
    # redirectable_test()
    test_im = "./test.png"
    import_img = cv2.imread(test_im, cv2.IMREAD_GRAYSCALE)
    time_1 = time()
    test_space = Atlas(import_img)
    print("Took {} seconds.".format(time() - time_1))
    print("{} black objects and {} white objects.".format(
        len(test_space[0]), len(test_space[1])))
    print(test_space)
    test_resegment(test_space)
    sys.exit(0)
