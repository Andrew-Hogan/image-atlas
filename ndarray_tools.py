"""Helps with numpy arrays!"""
import numpy as np
import cv2

import import_tools


DEFAULT_IMAGE_THRESH = 122


def np_shift(arr, num, fill_value=None):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result


def np_shift_v(arr, num, fill_value=None):
    result = np.empty_like(arr)
    if num > 0:
        result[:num, :] = fill_value
        result[num:, :] = arr[:-num, :]
    elif num < 0:
        result[num:, :] = fill_value
        result[:num, :] = arr[-num:, :]
    else:
        result = arr
    return result


def np_shift_h(arr, num, fill_value=None):
    result = np.empty_like(arr)
    if num > 0:
        result[:, :num] = fill_value
        result[:, num:] = arr[:, :-num]
    elif num < 0:
        result[:, num:] = fill_value
        result[:, :num] = arr[:, -num:]
    else:
        result = arr
    return result


def import_image_as_ndarray(image_or_image_file, grayscale=True):
    if isinstance(image_or_image_file, str):
        return import_tools.import_any_image_file(
            image_or_image_file, im_mode=import_tools.LOAD_CV2_GRAYSCALE if grayscale else None
        )
    elif isinstance(image_or_image_file, np.ndarray):
        if grayscale:
            return format_ndarray_dimension_to_grayscale(image_or_image_file)
        return image_or_image_file
    elif grayscale:
        return cv2.cvtColor(np.array(image_or_image_file), cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(np.array(image_or_image_file), cv2.COLOR_RGB2BGR)


def format_ndarray_dimension_to_grayscale(image):
    if len(image.shape) > 2:
        try:
            return np.squeeze(image, axis=2)
        except ValueError:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def threshold_image(image, threshold=DEFAULT_IMAGE_THRESH, max_value=255):
    """Threshold an image using cv2 binary thresh."""
    _, converted_img = cv2.threshold(image, thresh=threshold, maxval=max_value, type=cv2.THRESH_BINARY)
    return converted_img


def auto_resize(image, square_size=32, *,
                fill_val=255,
                interp=cv2.INTER_AREA):
    (height, width) = image.shape

    if height > width:
        differ = height
        if height < square_size:
            interp = cv2.INTER_CUBIC
    else:
        differ = width
        if width < square_size:
            interp = cv2.INTER_CUBIC
    image = image * fill_val

    mask = np.full((differ, differ), fill_val, dtype=np.float64)
    x_pos = int((differ - width) / 2)
    y_pos = int((differ - height) / 2)
    mask[y_pos:y_pos + height, x_pos:x_pos + width] = image[0:height, 0:width]

    image = cv2.resize(mask, (square_size, square_size), interpolation=interp)
    image = image / fill_val
    return image


def rotate_image(image, angle):  # This is in limbo - needs to be looked at.
    """Rotate an image using an angle, used in correcting sloped column group characters for neural network."""
    (h, w) = image.shape
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    #if len(image.shape) > 2:
    #    result = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_AREA,
    #                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    #    result = result[:, :, 0]       |
    #else: -removed rest and moved down V

    # perform the actual rotation and return the image
    ##result = cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(255))

    result = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    result = threshold_image(result)  # threshold=threshold

    #minispace = ImageSpace(result)
    #minispace.create_shape_bboxes_and_contained_shape_references()
    #relevant_object = minispace.largest_blackspace
    #cropped = relevant_object.create_neural_network_input_for_shape()
    #return cropped
    return result
