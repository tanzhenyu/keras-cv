import numpy as np
import tensorflow as tf
from kerascv.layers.anchor_generators.anchor_generator import AnchorGenerator
from kerascv.layers.losses.ssd_loss_layer import SSDLossLayer
from kerascv.layers.matchers.greedy_bipartite import target_assign_tf_func


def test_single_gt_best_match():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300),
        scales=[0.2],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=True,
    )
    anchors = anchor_gen((2, 2))
    ground_truth_boxes = tf.constant([[0.14, 0.64, 0.34, 0.84]])
    ground_truth_labels = tf.constant([[8]], dtype=tf.int64)
    matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask = target_assign_tf_func(
        ground_truth_boxes, ground_truth_labels, anchors
    )
    matched_gt_boxes = tf.expand_dims(matched_gt_boxes, axis=0)
    anchors = tf.expand_dims(anchors, axis=0)
    positive_mask = tf.expand_dims(positive_mask, axis=0)
    negative_mask = tf.expand_dims(negative_mask, axis=0)
    loss_layer = SSDLossLayer()
    anchor_labels = tf.constant([[
        [0.2, .8, 0.], [.2, .2, .2], [.1, .1, .1], [.1, .1, .1]
    ]], dtype=tf.float32)
    gt_labels = tf.constant([[0, 2, 0, 0]], dtype=tf.int64)
    _ = loss_layer(matched_gt_boxes, anchors, gt_labels, anchor_labels, positive_mask, negative_mask)
    expected_reg_losses = tf.keras.losses.huber(matched_gt_boxes, anchors)
    expected_cls_losses = tf.keras.losses.sparse_categorical_crossentropy(gt_labels, anchor_labels)
    reg_losses, cls_losses = loss_layer.losses
    np.testing.assert_allclose(tf.reduce_sum(expected_reg_losses), reg_losses)
    np.testing.assert_allclose(tf.reduce_sum(expected_cls_losses), cls_losses)
