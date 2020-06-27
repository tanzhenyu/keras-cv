import tensorflow as tf
from kerascv.layers.losses.hard_neg_miner import HardNegativeMining


class SSDLossLayer(tf.keras.layers.Layer):
    """Defines the Loss layer for Single Shot Detector."""

    def __init__(
        self,
        alpha=1.0,
        negative_positive_ratio=3,
        minimum_negative_examples=0,
        from_logits=False,
        name=None,
        **kwargs
    ):
        self.alpha = alpha
        self.negative_positive_ratio = negative_positive_ratio
        self.minimum_negative_examples = minimum_negative_examples
        self.hard_negative_miner = HardNegativeMining(
            negative_positive_ratio, minimum_negative_examples
        )
        # self.reg_loss_fn = tf.keras.losses.Huber(
        #     reduction=tf.keras.losses.Reduction.NONE
        # )
        self.reg_loss_fn = self.smooth_L1_loss
        self.cls_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE
        )
        super(SSDLossLayer, self).__init__(name=name, **kwargs)

    # [batch_size, n_boxes, 4]
    # THERE IS A BUG IN keras.losses.Huber, WHERE IT AVERAGES THE LOSS ACROSS n_boxes
    # USING CUSTOMIZED LOSS INSTEAD
    def smooth_L1_loss(self, y_true, y_pred):
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
        # [batch_size, n_boxes]
        return tf.reduce_sum(l1_loss, axis=-1)

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
        reg_losses = tf.reduce_sum(reg_losses * tf.cast(positive_mask, reg_losses.dtype), axis=-1)
        reg_losses = tf.constant(self.alpha, dtype=reg_losses.dtype) * reg_losses
        n_positives = tf.reduce_sum(positive_mask)
        n_positives = tf.maximum(
            tf.constant(1, dtype=cls_losses.dtype),
            tf.cast(n_positives, dtype=cls_losses.dtype),
        )
        reg_losses = tf.reduce_sum(reg_losses) / n_positives
        self.add_loss(reg_losses)
        cls_losses = tf.reduce_sum(cls_losses) / n_positives
        self.add_loss(cls_losses)
        self.add_metric(reg_losses, name='reg_losses_metrics')
        self.add_metric(cls_losses, name='cls_losses_metrics')
        return y_reg_pred, y_cls_pred

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "negative_positive_ratio": self.negative_positive_ratio,
            "minimum_negative_examples": self.minimum_negative_examples,
        }
        base_config = super(SSDLossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
