import numpy as np
import tensorflow as tf


mean_color = np.asarray([123, 117, 104])


def random_flip_horizontal(image, ground_truth_boxes, prob=0.5):
    if tf.random.uniform(shape=()) > prob:
        # Normalized ground truth box coordinates.
        width = tf.constant(1.0, tf.float32)
        image = tf.image.flip_left_right(image)
        y_min, x_min, y_max, x_max = tf.split(
            ground_truth_boxes, num_or_size_splits=4, axis=-1
        )
        flipped_x_min = width - x_max
        flipped_x_max = width - x_min
        flipped_ground_truth_boxes = tf.concat(
            [y_min, flipped_x_min, y_max, flipped_x_max], axis=1
        )
        ground_truth_boxes = flipped_ground_truth_boxes
    return image, ground_truth_boxes


def normalize_ground_truth_boxes(image, ground_truth_boxes):
    height = tf.cast(image.shape[0], ground_truth_boxes.dtype)
    width = tf.cast(image.shape[1], ground_truth_boxes.dtype)
    y_min, x_min, y_max, x_max = tf.split(
        ground_truth_boxes, num_or_size_splits=4, axis=-1
    )
    y_min = tf.cast(y_min / height, tf.float32)
    y_max = tf.cast(y_max / height, tf.float32)
    x_min = tf.cast(x_min / width, tf.float32)
    x_max = tf.cast(x_max / width, tf.float32)
    return image, tf.concat([y_min, x_min, y_max, x_max], axis=-1)


def denormalize_ground_truth_boxes(image, ground_truth_boxes):
    height = tf.cast(image.shape[0], ground_truth_boxes.dtype)
    width = tf.cast(image.shape[1], ground_truth_boxes.dtype)
    y_min, x_min, y_max, x_max = tf.split(
        ground_truth_boxes, num_or_size_splits=4, axis=-1
    )
    y_min = tf.cast(y_min * height, tf.int64)
    y_max = tf.cast(y_max * height, tf.int64)
    x_min = tf.cast(x_min * width, tf.int64)
    x_max = tf.cast(x_max * width, tf.int64)
    return image, tf.concat([y_min, x_min, y_max, x_max], axis=-1)


def random_expand(image, ground_truth_boxes, prob=0.5, min_scale=1.0, max_scale=4.0):
    # expand this into a canvas, and put the image somewhere on the canvas
    # assume that ground_truth_boxes are denormalized.
    # assume image are tf.uint8.
    if tf.random.uniform(shape=()) > prob:
        img_height = tf.cast(image.shape[0], tf.float32)
        img_width = tf.cast(image.shape[1], tf.float32)
        scale = tf.random.uniform(shape=(), minval=min_scale, maxval=max_scale, dtype=tf.float32)
        canvas_height = tf.cast(scale * img_height, tf.float32)
        canvas_width = tf.cast(scale * img_width, tf.float32)
        pad_height = tf.cast(canvas_height - img_height, tf.int64)
        pad_width = tf.cast(canvas_width - img_width, tf.int64)
        pad_top = tf.random.uniform(shape=(), minval=0, maxval=pad_height, dtype=tf.int64)
        pad_bottom = pad_height - pad_top
        pad_left = tf.random.uniform(shape=(), minval=0, maxval=pad_width, dtype=tf.int64)
        pad_right = pad_width - pad_left

        image_r, image_g, image_b = tf.split(image, num_or_size_splits=3, axis=-1)
        mean_r = tf.cast(mean_color[0], tf.uint8)
        canvas_r = tf.pad(
            image_r,
            paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            mode="CONSTANT",
            constant_values=mean_r
        )
        mean_g = tf.cast(mean_color[0], tf.uint8)
        canvas_g = tf.pad(
            image_g,
            paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            mode="CONSTANT",
            constant_values=mean_g
        )
        mean_b = tf.cast(mean_color[0], tf.uint8)
        canvas_b = tf.pad(
            image_b,
            paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
            mode="CONSTANT",
            constant_values=mean_b
        )
        canvas = tf.concat([canvas_r, canvas_g, canvas_b], axis=-1)
        y_min, x_min, y_max, x_max = tf.split(ground_truth_boxes, num_or_size_splits=4, axis=-1)
        padded_y_min = y_min + pad_top
        padded_x_min = x_min + pad_left
        padded_y_max = y_max + pad_top
        padded_x_max = x_max + pad_left
        canvas_ground_truth_boxes = tf.concat([padded_y_min, padded_x_min, padded_y_max, padded_x_max], axis=-1)
        return canvas, canvas_ground_truth_boxes

    else:
        return image, ground_truth_boxes


