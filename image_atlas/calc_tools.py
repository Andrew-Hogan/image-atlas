"""Helps with calculating!"""
import math


def find_center_cluster(coordinates):
    """Average a group of (int, int) coordinates and truncate to (int, int)."""
    assert coordinates, "No coordinates in cluster - average is impossible."
    if not coordinates:
        return None, None
    average_x = 0
    average_y = 0
    for x, y in coordinates:
        average_x += x
        average_y += y
    average_x //= len(coordinates)
    average_y //= len(coordinates)

    return average_x, average_y


def is_inside(x1, y1, x2, y2, x3, y3, x4, y4):
    """Return True if bounding box one of shape (min_x, min_y, max_x, max_y) is entirely inside bounding box two."""
    inside = False
    if x3 <= x1:
        if y3 <= y1:
            if x4 >= x2:
                if y4 >= y2:
                    inside = True
    return inside


def find_dimension_from_box(vertical_minimum, vertical_maximum, horizontal_minimum, horizontal_maximum):
    return vertical_maximum - vertical_minimum, horizontal_maximum - horizontal_minimum


def find_coordinates_from_box(vertical_minimum, vertical_maximum, horizontal_minimum, horizontal_maximum):
    """Find center coordinate of bounding box."""
    vertical_location = abs(vertical_minimum + vertical_maximum) / 2
    horizontal_location = abs(horizontal_minimum + horizontal_maximum) / 2
    return vertical_location, horizontal_location


def find_distance(vertical_location_a, horizontal_location_a, vertical_location_b, horizontal_location_b):
    """Find distance between two points."""
    vertical_distance = vertical_location_b - vertical_location_a
    horizontal_distance = horizontal_location_b - horizontal_location_a
    distance = math.sqrt((vertical_distance * vertical_distance) + (horizontal_distance * horizontal_distance))
    return distance


def distance_from_two_point_line(x0, y0, x1, y1, x2, y2):  # p0 is the point  # distance_from_twopoint_line
    """Find point's distance from 2-point line."""
    nom = abs(((y2 - y1) * x0) - ((x2 - x1) * y0) + (x2 * y1) - (y2 * x1))
    denom = math.sqrt(pow((y2 - y1), 2) + pow((x2 - x1), 2))
    result = (nom / denom)
    return result
