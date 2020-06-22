import tensorflow as tf


def random_flip_horizontal(image, ground_truth_boxes):
    if tf.random.uniform(shape=()) > 0.2:
        # Normalized ground truth box coordinates.
        width = tf.constant(1., tf.float32)
        image = tf.image.flip_left_right(image)
        y_min, x_min, y_max, x_max = tf.split(ground_truth_boxes, num_or_size_splits=4, axis=-1)
        flipped_x_min = width - x_max
        flipped_x_max = width - x_min
        flipped_ground_truth_boxes = tf.concat([y_min, flipped_x_min, y_max, flipped_x_max], axis=1)
        ground_truth_boxes = flipped_ground_truth_boxes
    return image, ground_truth_boxes
