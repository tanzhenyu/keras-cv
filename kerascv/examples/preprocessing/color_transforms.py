import tensorflow as tf


def photometric_transform(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    if tf.random.uniform() > 0.5:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.rgb_to_hsv(image)
    if tf.random.uniform() > 0.5:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    if tf.random.uniform() > 0.5:
        image = tf.image.random_hue(image, 0.05)
    image = tf.image.hsv_to_rgb(image)
    # Ignore lightning noise for now.
    return image

