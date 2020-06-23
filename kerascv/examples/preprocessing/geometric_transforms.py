import numpy as np
import tensorflow as tf


mean_color = np.asarray([123, 117, 104])


def random_flip_horizontal(image, ground_truth_boxes, prob=0.5, normalized=True):
    with tf.name_scope('random_flip_horizontal'):
        if tf.random.uniform(shape=(), name='prob') > prob:
            if normalized:
                width = tf.constant(1.0, tf.float32)
            else:
                width = tf.cast(tf.shape(image)[1], ground_truth_boxes.dtype)
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
    with tf.name_scope('normalize_ground_truth_boxes'):
        height = tf.cast(tf.shape(image)[0], ground_truth_boxes.dtype)
        width = tf.cast(tf.shape(image)[1], ground_truth_boxes.dtype)
        y_min, x_min, y_max, x_max = tf.split(
            ground_truth_boxes, num_or_size_splits=4, axis=-1
        )
        y_min = tf.cast(y_min / height, tf.float32)
        y_max = tf.cast(y_max / height, tf.float32)
        x_min = tf.cast(x_min / width, tf.float32)
        x_max = tf.cast(x_max / width, tf.float32)
        return image, tf.concat([y_min, x_min, y_max, x_max], axis=-1)


def denormalize_ground_truth_boxes(image, ground_truth_boxes):
    with tf.name_scope('denormalize_ground_truth_boxes'):
        height = tf.cast(tf.shape(image)[0], ground_truth_boxes.dtype)
        width = tf.cast(tf.shape(image)[1], ground_truth_boxes.dtype)
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
    with tf.name_scope('random_expand'):
        if tf.random.uniform(shape=(), name='prob') > prob:
            img_height = tf.cast(tf.shape(image)[0], tf.float32)
            img_width = tf.cast(tf.shape(image)[1], tf.float32)
            scale = tf.random.uniform(shape=(), minval=min_scale, maxval=max_scale, dtype=tf.float32, name='scale')
            canvas_height = tf.cast(scale * img_height, tf.float32)
            canvas_width = tf.cast(scale * img_width, tf.float32)
            pad_height = tf.cast(canvas_height - img_height, tf.int64)
            pad_width = tf.cast(canvas_width - img_width, tf.int64)
            pad_top = tf.random.uniform(shape=(), minval=0, maxval=pad_height, dtype=tf.int64, name='pad_top')
            pad_bottom = pad_height - pad_top
            pad_left = tf.random.uniform(shape=(), minval=0, maxval=pad_width, dtype=tf.int64, name='pad_left')
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
            canvas.set_shape((None, None, 3))
            return canvas, canvas_ground_truth_boxes

        else:
            return image, ground_truth_boxes


def iou_numpy(box_a, box_b):
    # box_a: [n_gt_boxes, 4], box_b: [4]
    max_yx = np.minimum(box_a[:, 2:], box_b[2:])
    min_yx = np.maximum(box_a[:, :2], box_b[:2])
    intersect_side = np.clip((max_yx - min_yx), a_min=0, a_max=np.inf)
    # [n_gt_boxes, 4]
    intersect_height = intersect_side[:, 0]
    intersect_width = intersect_side[:, 1]
    intersect = intersect_height * intersect_width
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersect
    return intersect / union


def random_patch_numpy(image, ground_truth_boxes, ground_truth_labels, lower_scale=0.3,
                       lower_aspect_ratio=0.5, upper_aspect_ratio=2.0):
    # the minimum and maximum iou threshold
    patch_choices = (
        # use the original image,
        None,
        # minimum iou threshold .1, .3, .5, .7, .9
        (.1, None),
        (.3, None),
        (.5, None),
        (.7, None),
        (.9, None),
        # randomly sample a patch
        (None, None))
    patch_choices_len = len(patch_choices)
    # assuming ground truth boxes are denormalized.
    img_height, img_width, _ = image.shape
    while True:
        mode_ind = np.random.choice(patch_choices_len)
        mode = patch_choices[mode_ind]
        if not mode:
            return image, ground_truth_boxes, ground_truth_labels

        iou_threshold, _ = mode
        if iou_threshold is None:
            iou_threshold = float('-inf')

        for _ in range(50):
            # randomly select crop height and width
            crop_height = np.random.uniform(lower_scale * img_height, img_height)
            crop_width = np.random.uniform(lower_scale * img_width, img_width)

            # condition 1: aspect ratio must be between lower and upper bound
            if crop_height / crop_width < lower_aspect_ratio or crop_height / crop_width > upper_aspect_ratio:
                continue

            # condition 2: at least one ground truth box has center inside the crop
            crop_y_min = np.random.uniform(img_height - crop_height)
            crop_x_min = np.random.uniform(img_width - crop_width)
            crop = np.array(
                [int(crop_y_min), int(crop_x_min), int(crop_y_min + crop_height), int(crop_x_min + crop_width)])
            gt_centers = (ground_truth_boxes[:, :2] + ground_truth_boxes[:, 2:]) / 2.0
            mask_min = (crop[0] < gt_centers[:, 0]) * (crop[1] < gt_centers[:, 1])
            mask_max = (crop[2] > gt_centers[:, 0] * (crop[3] > gt_centers[:, 1]))
            mask = mask_min * mask_max
            if not mask.any():
                continue

            # condition 3: after condition 2, at least one ground truth box has iou with the random crop larger
            # than threshold the relative y_min to the original image
            filtered_gt_boxes = ground_truth_boxes[mask, :].copy()
            iou = iou_numpy(filtered_gt_boxes, crop)
            if iou.min() < iou_threshold:
                continue

            cropped_image = image[crop[0]:crop[2], crop[1]:crop[3], :]
            filtered_gt_labels = ground_truth_labels[mask]
            filtered_gt_boxes[:, :2] = np.maximum(filtered_gt_boxes[:, :2], crop[:2])
            filtered_gt_boxes[:, :2] -= crop[:2]
            filtered_gt_boxes[:, 2:] = np.minimum(filtered_gt_boxes[:, 2:], crop[2:])
            filtered_gt_boxes[:, 2:] -= crop[:2]

            return cropped_image, filtered_gt_boxes, filtered_gt_labels


def random_patch_tf(image, ground_truth_boxes, ground_truth_labels):
    patched_image, patched_ground_truth_boxes, patched_ground_truth_labels = tf.numpy_function(
        func=random_patch_numpy,
        inp=[image, ground_truth_boxes, ground_truth_labels],
        Tout=[tf.uint8, tf.int64, tf.int64]
    )
    patched_image.set_shape([None, None, 3])
    patched_ground_truth_boxes.set_shape([None, 4])
    patched_ground_truth_labels.set_shape([None])
    return patched_image, patched_ground_truth_boxes, patched_ground_truth_labels
