import numpy as np
import tensorflow as tf
from kerascv.layers.anchor_generators.anchor_generator import AnchorGenerator
from kerascv.layers.matchers.greedy_bipartite import target_assign_func
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
    ground_truth_labels = tf.constant([[8]])
    matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask = target_assign_func(
        ground_truth_boxes, ground_truth_labels, anchors
    )
    expected_matched_gt_boxes = np.asarray(
        [anchors[0, :], ground_truth_boxes[0, :], anchors[2, :], anchors[3, :]]
    )
    np.testing.assert_allclose(expected_matched_gt_boxes, matched_gt_boxes)
    expected_matched_gt_labels = np.zeros((4, 1))
    expected_matched_gt_labels[1] = ground_truth_labels[0]
    np.testing.assert_allclose(expected_matched_gt_labels, matched_gt_labels)
    expected_positive_mask = np.asarray([0, 1, 0, 0]).astype(np.int)
    expected_negative_mask = np.asarray([1, 0, 1, 1]).astype(np.int)
    np.testing.assert_equal(expected_positive_mask, positive_mask)
    np.testing.assert_equal(expected_negative_mask, negative_mask)


def test_single_gt_no_intersect():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300),
        scales=[0.2],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=True,
    )
    anchors = anchor_gen((2, 2))
    ground_truth_boxes = tf.constant([[0.4, 0.65, 0.6, 0.85]])
    ground_truth_labels = tf.constant([[8]])
    # Since it does not intersect with any anchor, it will be matched with the first gt.
    matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask = target_assign_func(
        ground_truth_boxes, ground_truth_labels, anchors
    )
    expected_matched_gt_boxes = np.asarray(
        [ground_truth_boxes[0, :], anchors[1, :], anchors[2, :], anchors[3, :]]
    )
    np.testing.assert_allclose(expected_matched_gt_boxes, matched_gt_boxes)
    expected_matched_gt_labels = np.zeros((4, 1))
    expected_matched_gt_labels[0] = ground_truth_labels[0]
    np.testing.assert_allclose(expected_matched_gt_labels, matched_gt_labels)
    expected_positive_mask = np.asarray([1, 0, 0, 0]).astype(np.int)
    expected_negative_mask = np.asarray([0, 1, 1, 1]).astype(np.int)
    np.testing.assert_equal(expected_positive_mask, positive_mask)
    np.testing.assert_equal(expected_negative_mask, negative_mask)


def test_single_gt_single_match_single_neutral():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300),
        scales=[0.5],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=True,
    )
    anchors = anchor_gen((2, 2))
    ground_truth_boxes = tf.constant([[0.24, 0.5, 0.74, 1.0]])
    ground_truth_labels = tf.constant([[8]])
    matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask = target_assign_func(
        ground_truth_boxes, ground_truth_labels, anchors
    )
    expected_matched_gt_boxes = np.asarray(
        [anchors[0, :], ground_truth_boxes[0, :], anchors[2, :], anchors[3, :]]
    )
    np.testing.assert_allclose(expected_matched_gt_boxes, matched_gt_boxes)
    expected_matched_gt_labels = np.zeros((4, 1))
    expected_matched_gt_labels[1] = ground_truth_labels[0]
    np.testing.assert_allclose(expected_matched_gt_labels, matched_gt_labels)
    expected_positive_mask = np.asarray([0, 1, 0, 0]).astype(np.int)
    expected_negative_mask = np.asarray([1, 0, 1, 0]).astype(np.int)
    np.testing.assert_equal(expected_positive_mask, positive_mask)
    np.testing.assert_equal(expected_negative_mask, negative_mask)


def test_single_gt_single_match_zero_neutral():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300),
        scales=[0.5],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=True,
    )
    anchors = anchor_gen((2, 2))
    ground_truth_boxes = tf.constant([[0.24, 0.5, 0.74, 1.0]])
    ground_truth_labels = tf.constant([[8]])
    matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask = target_assign_func(
        ground_truth_boxes, ground_truth_labels, anchors, negative_iou_threshold=1 / 3
    )
    expected_matched_gt_boxes = np.asarray(
        [anchors[0, :], ground_truth_boxes[0, :], anchors[2, :], anchors[3, :]]
    )
    np.testing.assert_allclose(expected_matched_gt_boxes, matched_gt_boxes)
    expected_matched_gt_labels = np.zeros((4, 1))
    expected_matched_gt_labels[1] = ground_truth_labels[0]
    np.testing.assert_allclose(expected_matched_gt_labels, matched_gt_labels)
    expected_positive_mask = np.asarray([0, 1, 0, 0]).astype(np.int)
    expected_negative_mask = np.asarray([1, 0, 1, 1]).astype(np.int)
    np.testing.assert_equal(expected_positive_mask, positive_mask)
    np.testing.assert_equal(expected_negative_mask, negative_mask)


def test_single_gt_four_match():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300),
        scales=[0.5],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=True,
    )
    anchors = anchor_gen((2, 2))
    ground_truth_boxes = tf.constant([[0.25, 0.25, 0.75, 0.75]])
    ground_truth_labels = tf.constant([[8]])
    matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask = target_assign_func(
        ground_truth_boxes,
        ground_truth_labels,
        anchors,
        positive_iou_threshold=1 / 7,
        negative_iou_threshold=1 / 8,
    )
    expected_matched_gt_boxes = np.tile(ground_truth_boxes, (4, 1))
    np.testing.assert_allclose(expected_matched_gt_boxes, matched_gt_boxes)
    expected_matched_gt_labels = np.tile(ground_truth_labels, (4, 1))
    np.testing.assert_allclose(expected_matched_gt_labels, matched_gt_labels)
    expected_positive_mask = np.asarray([1, 1, 1, 1]).astype(np.int)
    expected_negative_mask = np.asarray([0, 0, 0, 0]).astype(np.int)
    np.testing.assert_equal(expected_positive_mask, positive_mask)
    np.testing.assert_equal(expected_negative_mask, negative_mask)


def test_single_gt_single_match_three_negative():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300),
        scales=[0.5],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=True,
    )
    anchors = anchor_gen((2, 2))
    ground_truth_boxes = tf.constant([[0.25, 0.25, 0.75, 0.75]])
    ground_truth_labels = tf.constant([[8]])
    matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask = target_assign_func(
        ground_truth_boxes, ground_truth_labels, anchors
    )
    expected_matched_gt_boxes = np.asarray(
        [ground_truth_boxes[0, :], anchors[1, :], anchors[2, :], anchors[3, :]]
    )
    np.testing.assert_allclose(expected_matched_gt_boxes, matched_gt_boxes)
    expected_matched_gt_labels = np.zeros((4, 1))
    expected_matched_gt_labels[0] = ground_truth_labels[0]
    np.testing.assert_allclose(expected_matched_gt_labels, matched_gt_labels)
    expected_positive_mask = np.asarray([1, 0, 0, 0]).astype(np.int)
    expected_negative_mask = np.asarray([0, 1, 1, 1]).astype(np.int)
    np.testing.assert_equal(expected_positive_mask, positive_mask)
    np.testing.assert_equal(expected_negative_mask, negative_mask)


def test_single_gt_single_match_three_neutral():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300),
        scales=[0.5],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=True,
    )
    anchors = anchor_gen((2, 2))
    ground_truth_boxes = tf.constant([[0.25, 0.25, 0.75, 0.75]])
    ground_truth_labels = tf.constant([[8]])
    matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask = target_assign_func(
        ground_truth_boxes, ground_truth_labels, anchors, negative_iou_threshold=1 / 7
    )
    expected_matched_gt_boxes = np.asarray(
        [ground_truth_boxes[0, :], anchors[1, :], anchors[2, :], anchors[3, :]]
    )
    np.testing.assert_allclose(expected_matched_gt_boxes, matched_gt_boxes)
    expected_matched_gt_labels = np.zeros((4, 1))
    expected_matched_gt_labels[0] = ground_truth_labels[0]
    np.testing.assert_allclose(expected_matched_gt_labels, matched_gt_labels)
    expected_positive_mask = np.asarray([1, 0, 0, 0]).astype(np.int)
    expected_negative_mask = np.asarray([0, 0, 0, 0]).astype(np.int)
    np.testing.assert_equal(expected_positive_mask, positive_mask)
    np.testing.assert_equal(expected_negative_mask, negative_mask)


def test_tf_single_gt_single_match_three_neutral():
    anchor_gen = AnchorGenerator(
        image_size=(300, 300),
        scales=[0.5],
        aspect_ratios=[1.0],
        clip_boxes=False,
        normalize_coordinates=True,
    )
    anchors = anchor_gen((2, 2))
    ground_truth_boxes = tf.constant([[0.25, 0.25, 0.75, 0.75]])
    ground_truth_labels = tf.constant([[8]], dtype=tf.int64)
    matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask = target_assign_tf_func(
        ground_truth_boxes,
        ground_truth_labels,
        anchors,
        negative_iou_threshold=tf.constant(1 / 7, dtype=tf.float32),
    )
    expected_matched_gt_boxes = np.asarray(
        [ground_truth_boxes[0, :], anchors[1, :], anchors[2, :], anchors[3, :]]
    )
    np.testing.assert_allclose(expected_matched_gt_boxes, matched_gt_boxes)
    expected_matched_gt_labels = np.zeros((4, 1))
    expected_matched_gt_labels[0] = ground_truth_labels[0]
    np.testing.assert_allclose(expected_matched_gt_labels, matched_gt_labels)
    expected_positive_mask = np.asarray([1, 0, 0, 0]).astype(np.int)
    expected_negative_mask = np.asarray([0, 0, 0, 0]).astype(np.int)
    np.testing.assert_equal(expected_positive_mask, positive_mask)
    np.testing.assert_equal(expected_negative_mask, negative_mask)
