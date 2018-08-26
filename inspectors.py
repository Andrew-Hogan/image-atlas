import numpy as np

import ndarray_tools

CIRCLE_RESEGMENT = "resegment_circle"
SQUARE_RESEGMENT = "resegment_squares"
TEXT_RESEGMENT = "resegment_text"
RESEGMENTABLES = {"resegment_circle": True, "resegment_squares": True, "resegment_text": False}
MIN_CIRCLE_PIXELS = 30
RADIUS_TOLERANCE = (0.75, 1.25)
RADIUS_KEY = "radii"
DEVIATION_KEY = "deviation"


class Lens(object):
    def __init__(self, atlas, *, grayscale=True, binary=True):
        self.atlas = atlas
        self.grayscale = grayscale
        self.binary = binary
        self.source_image = None
        self.rows = None
        self.columns = None

    def four_connected_format_image(self, image, image_threshold):
        if self.binary:
            converted = ndarray_tools.threshold_image(image, threshold=image_threshold, max_value=1)
            self.rows = len(converted)
            self.columns = len(converted[0])
            return converted
        raise NotImplementedError("Sorry, non-binary image processing hasn't been added to this module!")

    def __call__(self, image_or_image_file, threshold=ndarray_tools.DEFAULT_IMAGE_THRESH):
        self.source_image = image_or_image_file
        return self.four_connected_format_image(
            ndarray_tools.import_image_as_ndarray(image_or_image_file,
                                                  grayscale=self.grayscale),
            image_threshold=threshold
        )


class Morphologist(object):
    def __init__(self, *_, **resegment_dict):
        self.resegment_dict = {}
        for resegment_key, should_resegment in RESEGMENTABLES.items():
            self.resegment_dict.update(resegment_dict.get(resegment_key, should_resegment))
        self.circle_average_radius = None
        self.circle_average_deviation = None

    def find_circles_in(self, owning_shape, color_key, *args, **kwargs):
        circles = {}
        first_pass_filtered = set()
        for shape in owning_shape.owned_of(color_key):
            first_pass_circle_check(shape, circles, first_pass_filtered)
        if not circles:
            return {}, set()
        self.circle_average_radius = np.average([stat_dict.get(RADIUS_KEY) for stat_dict in circles.values()])
        self.circle_average_deviation = np.average([stat_dict.get(DEVIATION_KEY) for stat_dict in circles.values()])
        if not self.resegment_dict.get(CIRCLE_RESEGMENT):
            return circles, set()
        self.resegment_from_circles(circles, first_pass_filtered, *args, **kwargs)

    def resegment_from_circles(self, circles, potential_circles,
                               minimum_pixels=MIN_CIRCLE_PIXELS,
                               radii_range=RADIUS_TOLERANCE):
        minimum_radius = radii_range[0] * (self.circle_average_radius - self.circle_average_deviation)
        maximum_radius = radii_range[1] * (self.circle_average_radius + self.circle_average_deviation)


    def __call__(self, *args, **kwargs):
        return self.find_circles_in(*args, **kwargs)


def first_pass_circle_check(shape, circles, filtered):
    if shape.owned_of(0) and shape.owned_of(1):
        is_circle, pixel_radii, pixel_deviation = inspect_as_circle(shape.pixels)
        if is_circle:
            circles.update({shape: {RADIUS_KEY: pixel_radii, DEVIATION_KEY: pixel_deviation}})
        else:
            filtered.add(shape)


def inspect_as_circle(pixels):
    return is_shape_circle(pixels), circle_statistics(pixels), 1


def circle_statistics(pixels):
    return 1, 2


def is_shape_circle(pixels):
    return True
