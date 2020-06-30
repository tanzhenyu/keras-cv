import numpy as np
import tensorflow as tf
from kerascv.layers.iou_similarity import IOUSimilarity

iou_layer = IOUSimilarity()


# This is not a layer yet, because we need numpy processing
# Use iou_threshold >=0.5 as positive anchor, and <0.3 as negative anchor
def _target_assign_func(
    similarity,
    ground_truth_boxes,
    ground_truth_labels,
    anchors,
    positive_iou_threshold=0.5,
    negative_iou_threshold=0.5,
):
    # ground_truth_boxes: [n_boxes, 4]
    # ground_truth_labels: [n_boxes, 1]
    # anchors: [n_anchors, 4]

    # First calculate iou similarity
    # [n_gt_boxes, n_anchors]
    weight_matrix = np.copy(similarity)
    num_gt_boxes = weight_matrix.shape[0]
    num_anchors = weight_matrix.shape[1]
    gt_indices = list(range(num_gt_boxes))
    matched_gt_boxes = np.copy(anchors)
    matched_gt_labels = np.zeros((num_anchors, 1), dtype=np.int64)
    positive_mask = np.zeros(num_anchors, dtype=np.int)
    negative_mask = np.ones(num_anchors, dtype=np.int)
    if ground_truth_boxes.size == 0:
        print('No object found')
        return matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask

    # keep a record of the best anchor index for each ground truth box
    matches = np.zeros(num_gt_boxes, dtype=np.int)

    # greedy bipartite matching
    for _ in range(num_gt_boxes):
        # [n_gt_boxes], for each ground truth, find the best anchor index with best
        # overlap
        anchor_indices = np.argmax(weight_matrix, axis=1)
        # [n_gt_boxes], get the list of best overlap value
        overlaps = weight_matrix[gt_indices, anchor_indices]
        # find the ground truth box index with the highest overlap
        ground_truth_index = np.argmax(overlaps)
        # find the will-be assigned anchor with the highest overlap
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index
        # positive anchor found
        positive_mask[anchor_index] = 1
        # not a negative anchor
        negative_mask[anchor_index] = 0
        # assign the ground truth box and label to the matched anchor
        matched_gt_boxes[anchor_index, :] = ground_truth_boxes[ground_truth_index, :]
        matched_gt_labels[anchor_index, :] = ground_truth_labels[ground_truth_index, :]
        # resetting weight matrix so the anchor index and ground truth index does not
        # show up in the next iteration, this is until the entire weight matrix
        # becomes 0
        weight_matrix[ground_truth_index, :] = -0.1
        weight_matrix[:, anchor_index] = -0.1

    # argmax matching
    # set the already matched anchors to negative so that it does not show up
    weight_matrix = np.copy(similarity)
    weight_matrix[:, matches] = -0.1

    anchor_indices = list(range(num_anchors))
    # [n_anchors], the best matched gt index for each anchor
    ground_truth_indices = np.argmax(weight_matrix, axis=0)
    # [n_anchors], the best iou for each anchor
    overlaps = weight_matrix[ground_truth_indices, anchor_indices]
    positive_anchor_indices = np.nonzero(overlaps >= positive_iou_threshold)[0]
    if len(positive_anchor_indices) > 0:
        positive_gt_indices = ground_truth_indices[positive_anchor_indices]
        # In case positive_gt_indices has duplicates, do for loop here
        for positive_anchor_index, positive_gt_index in zip(
            positive_anchor_indices, positive_gt_indices
        ):
            matched_gt_boxes[positive_anchor_index, :] = ground_truth_boxes[
                positive_gt_index, :
            ]
            matched_gt_labels[positive_anchor_index, :] = ground_truth_labels[
                positive_gt_index, :
            ]
        positive_mask[positive_anchor_indices] = 1
        negative_mask[positive_anchor_indices] = 0
        # set the new matched anchors to 0 for filtering out neutral boxes
        weight_matrix[:, positive_anchor_indices] = 0.0

    # now weight matrix for all positive anchors are 0, let's find neutral anchors
    max_overlaps = np.amax(weight_matrix, axis=0)
    neutral_anchor_indices = np.nonzero(max_overlaps >= negative_iou_threshold)[0]
    negative_mask[neutral_anchor_indices] = 0

    return matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask


def target_assign_func(
    ground_truth_boxes,
    ground_truth_labels,
    anchors,
    positive_iou_threshold=0.5,
    negative_iou_threshold=0.3,
):
    similarity = iou_layer(ground_truth_boxes, anchors)
    return _target_assign_func(
        similarity,
        ground_truth_boxes,
        ground_truth_labels,
        anchors,
        positive_iou_threshold,
        negative_iou_threshold,
    )


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ]
)
def target_assign_tf_func(
    ground_truth_boxes,
    ground_truth_labels,
    anchors,
    positive_iou_threshold=tf.constant(0.5, dtype=tf.float32),
    negative_iou_threshold=tf.constant(0.5, dtype=tf.float32),
):
    # do not calculate similarity in numpy functions.
    similarity = iou_layer(ground_truth_boxes, anchors)
    return tf.numpy_function(
        _target_assign_func,
        [
            similarity,
            ground_truth_boxes,
            ground_truth_labels,
            anchors,
            positive_iou_threshold,
            negative_iou_threshold,
        ],
        [tf.float32, tf.int64, tf.int64, tf.int64],
    )
