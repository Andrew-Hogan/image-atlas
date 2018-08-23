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


def find_center_of_bounding_box(x1, y1, x2, y2):
    """Find center coordinate of bounding box."""
    x = abs(x1 + x2) / 2
    y = abs(y1 + y2) / 2
    return x, y


def find_distance(x1, y1, x2, y2):
    """Find distance between two points."""
    x_dis = x2 - x1
    y_dis = y2 - y1
    dis = math.sqrt((x_dis * x_dis) + (y_dis * y_dis))
    return dis


def distance_from_two_point_line(x0, y0, x1, y1, x2, y2):  # p0 is the point  # distance_from_twopoint_line
    """Find point's distance from 2-point line."""
    nom = abs(((y2 - y1) * x0) - ((x2 - x1) * y0) + (x2 * y1) - (y2 * x1))
    denom = math.sqrt(pow((y2 - y1), 2) + pow((x2 - x1), 2))
    result = (nom / denom)
    return result


