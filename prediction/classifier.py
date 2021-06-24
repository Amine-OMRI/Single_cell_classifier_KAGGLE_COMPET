"""
A class that initialize and allows to predict from three elements :
- a grayscale image representing the interest pixels of an observation through a microscope.
- a mask segmenting the distinct cells by respective values.
- a mask segmenting the distinct nuclei by respective values, also respective to their cell.

Accepts any size of images and mask, as long as the three of them have the same shape.

Limitation : as mask tensors have a type fixed at uint8, a mask cannot hold more than 255 classes (0 being void),
meaning more than 255 cells.
"""

import numpy as np
import tensorflow as tf
from typing import Generator, Any
import os

import config

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Allow_Growth on GPU enabled.")
except:
    print("Not able to enable allow_growth.")

try:
    print("Loading default classifier...", end="")
    model = tf.keras.models.load_model(os.path.join("prediction", "models", config.MODEL_PATH))
except OSError as e:
    print("Could not load the model with direct relative path.")
    print(e)
    print("Trying to load from project location...", end="")
    model = tf.keras.models.load_model(os.path.join(config.APP_NAME, "prediction", "models", config.MODEL_PATH),
                                       custom_objects={'tf': tf, "BATCH_SIZE": config.BATCH_SIZE})
print("done.")
print("Model loaded successfully.")


def __unpad(tf_mask: tf.Tensor) -> list:
    """
    Calculates the smallest rectangle that can hold all the True values in a 2D tensor,
    also known as "bounding box coordinates".

    :param tf_mask: Tensor to analyze.
    :type tf_mask: tensorflow.Tensor (dtype=tf.uint8, ndims=2)
    :return: A list of 4 coordinates :
        - first row holding a True
        - last row to hod a True
        - first column holding a True
        - last column holding a True
    :rtype: list[int, int, int, int]
    """

    rows = tf.where(tf.reduce_any(tf_mask, 1)).numpy()[0]
    cols = tf.where(tf.reduce_any(tf_mask, 0)).numpy()[0]
    rmin, rmax = rows[0], rows[-1]
    cmin, cmax = cols[0], cols[-1]

    return [rmin, rmax, cmin, cmax]


def __get_class_bbox(cell_mask: tf.Tensor, class_idx: int, pads: tuple = None, padding: float = 0.01) -> list:
    f"""
    Extracts the bounding box coordinates of a specific value in a tensor and adds padding.
    
    :param cell_mask: Mask from which to extract the bounding box coordinates.
    :type cell_mask: tensorflow.Tensor (dtype=tf.uint8, ndims=2)
    :param class_idx: Value to look for in the [cell_mask].
    :type class_idx: int
    :param pads: Padding to apply to the bounding box. If set to None, the value will be inferred from {padding}.
    :type pads: tuple of 4 coordinates : x padding left, x padding right, y padding top, y padding bottom.
    :param padding: Ratio of padding to add, relative to the [cell_mask] tensor size. Ignored if pads != None.
    :return: A list of 4 coordinates :
        - x start of the bounding box
        - x end of the bounding box
        - y start of the bounding box
        - y end of the bounding box.
    :rtype: list[int, int, int, int]
    """

    h, w = tf.shape(cell_mask)

    if pads is None:
        pads = int(int(h) * padding), int(int(w) * padding)

    bb = __unpad(tf.equal(cell_mask, class_idx))  # Not using operator == for compatibility issues

    bb[0] = max(bb[0] - pads[0], 0)
    bb[1] = min(bb[1] + pads[0], h)
    bb[2] = max(bb[2] - pads[1], 0)
    bb[3] = min(bb[3] + pads[1], w)

    return bb


def __getall_bboxes(cell_mask: tf.Tensor, padding: float = 0.01) -> Generator[list, Any, None]:
    """
    Returns all the bounding boxes of a mask.
    The count of bounding boxes equals the maximum value inside the [cell_mask].
    If a value is missing, will raise an Exception.

    :param cell_mask: Tensor representing a mask.
    :type cell_mask: tensorflow.Tensor (dtype=tf.uint8, ndims=2)
    :param padding: Ratio of padding to add, relative to the [cell_mask] tensor size.
    :return: A generator yielding a bbox per class (value except zero) present in the [cell_mask].
    :rtype: generator
    """

    pads = int(int(tf.shape(cell_mask)[0]) * padding), int(int(tf.shape(cell_mask)[1]) * padding)
    return (__get_class_bbox(cell_mask, i + 1, pads) for i in range(np.max(cell_mask.numpy())))


def __slice_crop(cell_mask: tf.Tensor, bbox: list, filter_in: int = None) -> tf.Tensor:
    """
    Returns a crop from an image and bbox coordinates.
    Can also binarize the crop by a specific class (value except zero).

    :param cell_mask: Mask to slice.
    :type cell_mask: tensorflow.Tensor (dtype=tf.uint8, ndims=2)
    :param bbox: Coordinates of the bounding box that will slice the [cell_mask].
    :type bbox: list[int, int, int, int]
    :param filter_in: Value ot filter_in. Each element equal to that value will become True, and the other will
    become False.
    :return: A smaller tensor, aka a crop.
    :rtype: tensorflow.Tensor
    """

    crop = cell_mask[bbox[0]:bbox[1], bbox[2]:bbox[3]]

    if filter_in:
        crop = tf.where(tf.equal(crop, filter_in), crop, 0)

    return crop


def __add_padding(crop: tf.Tensor, size: int):
    """
    Returns a tensor with added border.
    Made to positively resize a Tensor, preserving the ratio and the original size.

    :param crop: A 3D tensor, aka an image, to resize.
    :type crop: tensorflow.Tensor (dtype=tf.uint, ndims=3)
    :param size: Wanted size. Will apply to width and height, aka a square base.
    :return: A larger tensor.
    :rtype: tensorflow.Tensor
    """

    h, w = crop.shape[:2]
    dy, dx = (size - h) // 2, (size - w) // 2
    ret = tf.pad(crop, ((dy, size - h - dy), (dx, size - w - dx), (0, 0)))

    return ret


def __resize_crop(crop: tf.Tensor, size: int):
    """
    Resize a 3D tensor to a fixed square. Respects the proportions, thus doesn't distort the image.
    If the tensor needs to be enlarged, it will add padding.
    If the tensor needs to be smaller, it will be resized with a bit of fast interpolation.

    :param crop: A 3D tensor, aka an image, to resize.
    :type crop: tensorflow.Tensor (dtype=tf.uint8, ndims=3)
    :param size: Wanted size. Will apply to width and height, aka a square base.
    :return: A resized tensor.
    :rtype: tensorflow.Tensor
    """

    shape = tf.shape(crop)[:2].numpy()

    if max(shape) == size:
        return crop
    if (shape > size).any():
        return tf.image.resize_with_pad(crop, size, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return __add_padding(crop, size)


def predict(green_channel: np.array, cell_mask: np.array, nuclei_mask: np.array) -> tuple:
    """
    Predicts with the saved and initialized tensorflow model, and returns the most probable class, as well as its
    confidence (probability).

    :param green_channel: Array of the "green" channel of the image.
    :type green_channel: np.array (dtype=np.uint8, ndims=2)
    :param cell_mask: Array of the cellular segments mask of the image.
    :type cell_mask: np.array (dtype=np.uint8, ndims=2)
    :param nuclei_mask: Array of the nuclear segments mask of the image. The nuclei values must be respective to their
    cell in the [cell_mask].
    :type nuclei_mask: np.array (dtype=np.uint8, ndims=2)
    :return: The most probable class as a string, and its confidence as a float between 0 and 1 (1 being certainty).
    :rtype: tuple[str, float]
    """

    green_tensor = tf.convert_to_tensor(green_channel, dtype=tf.uint8)
    cell_tensor = tf.convert_to_tensor(cell_mask, dtype=tf.uint8)
    nuclei_tensor = tf.convert_to_tensor(nuclei_mask, dtype=tf.uint8)

    bboxes = __getall_bboxes(cell_tensor)

    cell_no_nuclei_tensor = tf.subtract(cell_tensor, nuclei_tensor)

    results = None

    for i, bbox in enumerate(bboxes):
        red_crop = __slice_crop(cell_no_nuclei_tensor, bbox, i + 1)
        blue_crop = __slice_crop(nuclei_tensor, bbox, i + 1)
        green_crop = __slice_crop(green_tensor, bbox)

        total_cell_crop = __slice_crop(cell_tensor, bbox)
        green_crop = tf.where(
            tf.math.logical_and(tf.greater(total_cell_crop, 0), tf.not_equal(total_cell_crop, i + 1)),
            0,
            green_crop)

        composite = tf.stack(
            (tf.multiply(red_crop, 255), green_crop, tf.multiply(blue_crop, 255)),
            2)
        composite = tf.dtypes.cast(__resize_crop(composite, config.CROP_SHAPE[0]), tf.float32)

        composite = tf.reshape(composite, (1, *config.CROP_SHAPE))
        prediction = model.predict(composite)
        results = prediction if results is None else tf.add(results, prediction)

    results = tf.math.softmax(tf.math.reduce_mean(results, 0))

    return config.ORGANELLE_CLASSES_SAMPLED[tf.argmax(results).numpy()], tf.reduce_max(results).numpy(),\
        dict(zip(config.ORGANELLE_CLASSES_SAMPLED, results.numpy()))
