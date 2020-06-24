import tensorflow as tf
from kerascv.layers.iou_similarity import IOUSimilarity

iou_layer = IOUSimilarity()


class ArgMaxMatcher(tf.keras.layers.Layer):
    """ArgMax matcher"""

    # [pos, neutral, neg]
    def __init__(self, matched_threshold, unmatched_threshold):
        self.matched_threshold = matched_threshold
        self.unmatched_threshold = unmatched_threshold
        super(ArgMaxMatcher, self).__init__()

    # similarity: [#num_anchors, #num_gt_boxes]
    # matched_values: [#num_gt_boxes, dim]
    # unmatched_values: [dim]
    # ignored_values: [dim]
    def call(self, similarity, matched_values, unmatched_values, ignored_values):
        # [#num_anchors]
        matched_indices = tf.argmax(similarity, axis=1)
        # [#num_anchors]
        matched_max_vals = tf.reduce_max(similarity, axis=1)
        above_unmatched_threshold_indices = tf.cast(
            tf.greater(matched_max_vals, self.unmatched_threshold), tf.float32
        )

        # [#num_anchors]
        below_unmatched_threshold_indices = tf.greater(
            self.unmatched_threshold, matched_max_vals
        )
        below_unmatched_threshold_indices = tf.cast(
            below_unmatched_threshold_indices, matched_values.dtype
        )
        # [#num_anchors]
        between_threshold_indices = tf.logical_and(
            tf.greater_equal(matched_max_vals, self.unmatched_threshold),
            tf.greater(self.matched_threshold, matched_max_vals),
        )
        between_threshold_indices = tf.cast(
            between_threshold_indices, matched_values.dtype
        )
        # [#num_anchors, dim]
        matched_vals = tf.gather(matched_values, matched_indices)
        if matched_vals.shape.rank > 1:
            # [#num_anchors, 1]
            below_unmatched_threshold_indices = below_unmatched_threshold_indices[
                :, tf.newaxis
            ]
            # [#num_anchors, 1]
            between_threshold_indices = between_threshold_indices[:, tf.newaxis]
        matched_vals = tf.add(
            tf.multiply(
                matched_vals,
                tf.constant(1, dtype=matched_values.dtype)
                - below_unmatched_threshold_indices,
            ),
            tf.multiply(unmatched_values, below_unmatched_threshold_indices),
        )
        matched_vals = tf.add(
            tf.multiply(
                matched_vals,
                tf.constant(1, dtype=matched_values.dtype) - between_threshold_indices,
            ),
            tf.multiply(ignored_values, between_threshold_indices),
        )
        return matched_vals

    def get_config(self):
        config = {
            "matched_threshold": self.matched_threshold,
            "unmatched_threshold": self.unmatched_threshold,
        }
        base_config = super(ArgMaxMatcher, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ]
)
def target_assign_argmax(
        ground_truth_boxes,
        ground_truth_labels,
        anchors,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3):
    # [n_gt_boxes, n_anchors]
    similarity = iou_layer(ground_truth_boxes, anchors)
    # [n_anchors]
    matched_gt_indices = tf.argmax(similarity, axis=0)
    # [n_anchors]
    matched_max_vals = tf.reduce_max(similarity, axis=0)
    positive_mask = tf.greater(matched_max_vals, positive_iou_threshold)
    negative_mask = tf.greater(negative_iou_threshold, matched_max_vals)
    matched_gt_boxes = tf.gather(ground_truth_boxes, matched_gt_indices)
    matched_gt_labels = tf.gather(ground_truth_labels, matched_gt_indices)
    return matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask