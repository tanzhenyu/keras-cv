import tensorflow as tf


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
