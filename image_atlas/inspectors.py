import math

import numpy as np

import ndarray_tools
import calc_tools


RADIUS_KEY = "radii"
DEVIATION_KEY = "deviation"
CIRCLE_RESEGMENT = "resegment_circle"
SQUARE_RESEGMENT = "resegment_squares"
TEXT_RESEGMENT = "resegment_text"
RESEGMENTABLES = {CIRCLE_RESEGMENT: True, SQUARE_RESEGMENT: True, TEXT_RESEGMENT: False}
MIN_CIRCLE_PIXELS = 30
RADIUS_TOLERANCE = (0.75, 1.25)
EDGE_SHAPE_DISTANCE_CUTOFF = 5
MAX_CIRCLE_PIXEL_MODIFIED_DEVIATION = .075


class Lens(object):
    def __init__(self, atlas, *, grayscale=True, binary=True):
        self.atlas = atlas
        self.grayscale = grayscale
        self.binary = binary
        self.source_image = None
        self.rows = None
        self.columns = None

    def format_image(self, image, image_threshold):
        if self.binary:
            converted = ndarray_tools.threshold_image(image, threshold=image_threshold, max_value=1)
            self.rows = len(converted)
            self.columns = len(converted[0])
            return converted
        raise NotImplementedError("Sorry, non-binary image processing hasn't been added to this module!")

    def __call__(self, image_or_image_file, threshold=ndarray_tools.DEFAULT_IMAGE_THRESH):
        self.source_image = image_or_image_file
        return self.format_image(ndarray_tools.import_image_as_ndarray(
            image_or_image_file, grayscale=self.grayscale), image_threshold=threshold
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
            is_circle, pixel_radii, pixel_deviation = first_pass_circle_check(shape)
            if is_circle:
                circles.update({shape: {RADIUS_KEY: pixel_radii, DEVIATION_KEY: pixel_deviation}})
            else:
                first_pass_filtered.add(shape)
        if not circles:
            return {}, set()
        self.circle_average_radius = np.average([stat_dict.get(RADIUS_KEY) for stat_dict in circles.values()])
        self.circle_average_deviation = np.average([stat_dict.get(DEVIATION_KEY) for stat_dict in circles.values()])
        if not self.resegment_dict.get(CIRCLE_RESEGMENT):
            return circles, set()
        self.resegment_from_circles(circles, first_pass_filtered, *args, **kwargs)

    def resegment_from_circles(self, circles, potential_circles, inner_shape_color_key,
                               minimum_pixels=MIN_CIRCLE_PIXELS,
                               radii_range=RADIUS_TOLERANCE):
        minimum_radius = radii_range[0] * (self.circle_average_radius - self.circle_average_deviation)
        maximum_radius = radii_range[1] * (self.circle_average_radius + self.circle_average_deviation)
        for shape in potential_circles:
            for owned_shape in shape.owned_of(inner_shape_color_key):
                is_circle = second_pass_circle_check(owned_shape, minimum_radius, maximum_radius)
                if is_circle:
                    pass  # This meatball is being extracted from some spaghetti. Please excuse the mess.

    def __call__(self, *args, **kwargs):
        return self.find_circles_in(*args, **kwargs)


def first_pass_circle_check(shape):
    if shape.owned_of(0) and shape.owned_of(1):
        is_circle, pixel_radii, pixel_deviation = inspect_as_circle(shape.pixels, shape.coordinates)
        return is_circle, pixel_radii, pixel_deviation
    return False, 0, 0


def inspect_as_circle(pixels, center_coordinates, max_modified_deviation=MAX_CIRCLE_PIXEL_MODIFIED_DEVIATION):
    pixel_radii, pixel_deviation = circle_statistics(pixels, center_coordinates)
    if pixel_radii:
        is_circle = is_shape_circle(pixel_radii, pixel_deviation, max_modified_deviation)
        return is_circle, pixel_radii, pixel_deviation
    return False, pixel_radii, pixel_deviation


def circle_statistics(pixels, center_coordinates):
    distances = pixel_distances_from_point(pixels, center_coordinates)
    if not distances:
        return 0, 0
    distances = np.array(distances)
    deviation = np.std(distances)
    average_distance = np.average(distances)
    return average_distance, deviation


def pixel_distances_from_point(pixels, coordinates):
    return [calc_tools.find_distance(pixel.coordinates, coordinates) for pixel in pixels]


def is_shape_circle(pixel_average_distance, pixel_deviation, max_modified_deviation):
    modified_deviation = pixel_deviation / pixel_average_distance
    if modified_deviation < max_modified_deviation:
        return True
    return False


def second_pass_circle_check(shape, minimum_radius, maximum_radius):
    distances = pixel_distances_from_point(shape.pixels, shape.coordinates)
    is_circular, maximum_distance = are_edge_distances_circle(distances)
    if is_circular and minimum_radius <= maximum_distance <= maximum_radius:
        return True
    return False


def are_edge_distances_circle(center_distances, edge_cutoff=EDGE_SHAPE_DISTANCE_CUTOFF):
    if not center_distances:
        return False, 0
    max_pixel_distance = max(center_distances)
    distance_cutoff = max_pixel_distance - edge_cutoff
    number_past_cutoff = len([pixel_distance for pixel_distance in center_distances
                              if pixel_distance > distance_cutoff])
    if number_past_cutoff > (distance_cutoff * math.pi * edge_cutoff):
        return True, max_pixel_distance
    return False, max_pixel_distance

def circle_subtraction(shape_pixels, center, average_radius, average_deviation, bounding_deviation_multiplier=2):
    maximum_distance = average_radius + abs(bounding_deviation_multiplier * average_deviation)
    minimum_distance = average_radius - abs(bounding_deviation_multiplier * average_deviation)

    pixels,
