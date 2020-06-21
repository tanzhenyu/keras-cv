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
    ground_truth_labels = tf.constant([[2]], dtype=tf.int64)
    matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask = target_assign_tf_func(
        ground_truth_boxes, ground_truth_labels, anchors
    )
    matched_gt_boxes = tf.expand_dims(matched_gt_boxes, axis=0)
    matched_gt_labels = tf.expand_dims(tf.squeeze(matched_gt_labels, axis=-1), axis=0)
    anchors = tf.expand_dims(anchors, axis=0)
    positive_mask = tf.expand_dims(positive_mask, axis=0)
    negative_mask = tf.expand_dims(negative_mask, axis=0)
    loss_layer = SSDLossLayer()
    anchor_labels = tf.constant([[
        [.2, .8, 0.], [.2, .2, .2], [.1, .1, .1], [.1, .1, .1]
    ]], dtype=tf.float32)
    _ = loss_layer(matched_gt_boxes, anchors, matched_gt_labels, anchor_labels, positive_mask, negative_mask)
    reg_losses, cls_losses = loss_layer.losses
    # reg loss is 0.5 * (0.01^2+0.01^2+0.01^2+0.01^2) = 0.0002
    expected_reg_losses = 0.0002
    np.testing.assert_allclose(expected_reg_losses, reg_losses, atol=1e-7)
    # cls loss is log(0.2) + log(1/3) + log(1/3) + log(1/3)
    expected_cls_losses = 4.905275
    np.testing.assert_allclose(expected_cls_losses, cls_losses)


def test_single_gt_best_match_from_logits():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300),
        scales=[0.2],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=True,
    )
    anchors = anchor_gen((2, 2))
    ground_truth_boxes = tf.constant([[0.14, 0.64, 0.34, 0.84]])
    ground_truth_labels = tf.constant([[2]], dtype=tf.int64)
    matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask = target_assign_tf_func(
        ground_truth_boxes, ground_truth_labels, anchors
    )
    matched_gt_boxes = tf.expand_dims(matched_gt_boxes, axis=0)
    matched_gt_labels = tf.expand_dims(tf.squeeze(matched_gt_labels, axis=-1), axis=0)
    anchors = tf.expand_dims(anchors, axis=0)
    positive_mask = tf.expand_dims(positive_mask, axis=0)
    negative_mask = tf.expand_dims(negative_mask, axis=0)
    loss_layer = SSDLossLayer(from_logits=True)
    anchor_labels = tf.constant([[
        [.2, .8, .0], [.2, .2, .2], [.1, .1, .1], [.1, .1, .1]
    ]], dtype=tf.float32)
    _ = loss_layer(matched_gt_boxes, anchors, matched_gt_labels, anchor_labels, positive_mask, negative_mask)
    reg_losses, cls_losses = loss_layer.losses
    # reg loss is 0.5 * (0.01^2+0.01^2+0.01^2+0.01^2) = 0.0002
    np.testing.assert_allclose(0.0002, reg_losses, atol=1e-7)
    # cls loss is (log(0.27466115) + log(1/3) + log(1/3) + log(1/3)) = 4.5880537
    expected_cls_losses = 4.5880537
    np.testing.assert_allclose(expected_cls_losses, cls_losses)


def test_two_gt_two_matches():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300),
        scales=[0.2],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=True,
    )
    anchors = anchor_gen((2, 2))
    print(anchors)
    # The first box will be matched to the second anchor
    # The second box will be matched to the first anchor
    ground_truth_boxes = tf.constant([
        [0.15, 0.65, 0.35, 0.85],
        [0.14, 0.64, 0.34, 0.84],
    ])
    ground_truth_labels = tf.constant([[1], [2]], dtype=tf.int64)
    matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask = target_assign_tf_func(
        ground_truth_boxes, ground_truth_labels, anchors
    )
    loss_layer = SSDLossLayer(from_logits=True)
    anchor_labels = tf.constant([[
        [.2, .8, .0], [.2, .2, .2], [.1, .1, .1], [.1, .1, .1]
    ]], dtype=tf.float32)
    matched_gt_boxes = tf.expand_dims(matched_gt_boxes, axis=0)
    matched_gt_labels = tf.expand_dims(tf.squeeze(matched_gt_labels, axis=-1), axis=0)
    anchors = tf.expand_dims(anchors, axis=0)
    positive_mask = tf.expand_dims(positive_mask, axis=0)
    negative_mask = tf.expand_dims(negative_mask, axis=0)
    _ = loss_layer(matched_gt_boxes, anchors, matched_gt_labels, anchor_labels, positive_mask, negative_mask)
    reg_losses, cls_losses = loss_layer.losses
    # reg loss is 0.5 * (0.01^2+0.49^2+0.01^2+0.49^2) / 2, since n_positives = 2
    np.testing.assert_allclose(0.12009999, reg_losses)
    # cls loss is (log(0.22487354) + log(1/3) + log(1/3) + log(1/3)) / 2, since n_positives = 2
    np.testing.assert_allclose(2.394027, cls_losses)
