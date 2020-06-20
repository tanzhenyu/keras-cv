import tensorflow as tf
from kerascv.layers.losses.hard_neg_miner import HardNegativeMining


class SSDLossLayer(tf.keras.layers.Layer):
    """Defines the Loss layer for Single Shot Detector."""

    def __init__(
        self,
        alpha=1.0,
        negative_positive_ratio=3,
        minimum_negative_examples=0,
        name=None,
        **kwargs
    ):
        self.alpha = alpha
        self.negative_positive_ratio = negative_positive_ratio
        self.minimum_negative_examples = minimum_negative_examples
        self.hard_negative_miner = HardNegativeMining(
            negative_positive_ratio, minimum_negative_examples
        )
        self.reg_loss_fn = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.cls_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
        super(SSDLossLayer, self).__init__(name=name, **kwargs)

    # y_reg_true, y_reg_pred: [batch_size, n_boxes, 4]
    # y_cls_true: [batch_size, n_boxes]
    # y_cls_pred: [batch_size, n_boxes, n_classes]
    # positive_mask, negative_mask: [batch_size, n_boxes]
    def call(
        self,
        y_reg_true,
        y_reg_pred,
        y_cls_true,
        y_cls_pred,
        positive_mask,
        negative_mask,
    ):
        cls_losses = self.cls_loss_fn(y_true=y_cls_true, y_pred=y_cls_pred)
        # [batch_size]
        cls_losses = self.hard_negative_miner(cls_losses, positive_mask, negative_mask)
        reg_losses = self.reg_loss_fn(y_true=y_reg_true, y_pred=y_reg_pred)
        # [batch_size]
        reg_losses = tf.reduce_sum(reg_losses, axis=1)
        reg_losses = tf.constant(self.alpha, dtype=reg_losses.dtype) * reg_losses
        n_positives = tf.reduce_sum(positive_mask)
        n_positives = tf.maximum(
            tf.constant(1, dtype=cls_losses.dtype),
            tf.cast(n_positives, dtype=cls_losses.dtype),
        )
        self.add_loss(tf.reduce_sum(reg_losses) / n_positives)
        self.add_loss(tf.reduce_sum(cls_losses) / n_positives)
        return y_reg_pred, y_cls_pred

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "negative_positive_ratio": self.negative_positive_ratio,
            "minimum_negative_examples": self.minimum_negative_examples,
        }
        base_config = super(SSDLossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
