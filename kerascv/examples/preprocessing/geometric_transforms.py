import tensorflow as tf


def random_flip(image, ground_truth_boxes):
    if tf.random.uniform(shape=()) > 0.5:
        image_shape = image.shape
        height = tf.cast(image_shape[0], dtype=tf.int64)
        width = tf.cast(image_shape[1], dtype=tf.int64)
        image = tf.image.flip_left_right(image)
        y_min, x_min, y_max, x_max = tf.split(ground_truth_boxes, num_or_size_splits=4, axis=-1)
        flipped_y_min = height - y_max
        flipped_y_max = height - y_min
        flipped_x_min = width - x_max
        flipped_x_max = width - x_min
        flipped_ground_truth_boxes = tf.concat([flipped_y_min, flipped_x_min, flipped_y_max, flipped_x_max], axis=1)
        ground_truth_boxes = flipped_ground_truth_boxes
    return image, ground_truth_boxes
