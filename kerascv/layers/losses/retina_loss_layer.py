import tensorflow as tf
from kerascv.layers.losses.hard_neg_miner import HardNegativeMining


class RetinaLossLayer(tf.keras.layers.Layer):
    """Defines the Loss layer for Single Shot Detector."""

    def __init__(
        self,
        n_classes,
        alpha=.25,
        gamma=2.0,
        name=None,
        **kwargs
    ):
        self.n_classes = n_classes
        self.alpha = alpha
        self.gamma = gamma
        # self.reg_loss_fn = tf.keras.losses.Huber(
        #     reduction=tf.keras.losses.Reduction.NONE
        # )
        self.reg_loss_fn = self.smooth_L1_loss
        self.cls_loss_fn = self.cls_loss
        super(RetinaLossLayer, self).__init__(name=name, **kwargs)

    # y_true [batch_size, n_boxes]
    # y_pred [batch_size, n_boxes, n_classes]
    def cls_loss(self, y_true, y_pred):
        alpha_t = tf.cast(self.alpha, tf.float32)
        gamma_t = tf.cast(self.gamma, tf.float32)
        ones = tf.constant(1., dtype=tf.float32)
        y_true = tf.one_hot(y_true, depth=self.n_classes, on_value=1.0, off_value=0.0)
        # [batch_size, n_boxes, n_classes]
        x_ent = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
        # run the sigmoid per class
        p = tf.nn.sigmoid(y_pred)
        positive_indices = tf.equal(y_true, ones)
        alpha = tf.where(positive_indices, alpha_t, (ones - alpha_t))
        pt = tf.where(positive_indices, p, ones - p)
        losses = alpha * tf.math.pow(ones - pt, gamma_t) * x_ent
        # [batch_size, n_boxes]
        return tf.reduce_sum(losses, axis=-1)

    # [batch_size, n_boxes, 4]
    # y_pred should ALWAYS come from logits directly
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
        # [batch_size, n_boxes]
        reg_losses = self.reg_loss_fn(y_true=y_reg_true, y_pred=y_reg_pred)
        positive_mask = tf.cast(positive_mask, reg_losses.dtype)
        negative_mask = tf.cast(negative_mask, reg_losses.dtype)
        n_positives = tf.reduce_sum(positive_mask)
        n_positives = tf.maximum(
            tf.constant(1, dtype=reg_losses.dtype),
            tf.cast(n_positives, dtype=reg_losses.dtype),
        )
        # regression loss includes positive anchors
        reg_losses = tf.reduce_sum(reg_losses * positive_mask) / n_positives
        self.add_loss(reg_losses)

        cls_losses = self.cls_loss_fn(y_true=y_cls_true, y_pred=y_cls_pred)
        # classification loss includes both positive and negative anchors
        pos_cls_losses = tf.reduce_sum(cls_losses * positive_mask) / n_positives
        neg_cls_losses = tf.reduce_sum(cls_losses * negative_mask) / n_positives
        self.add_metric(pos_cls_losses, name='pos_cls_loss')
        self.add_metric(neg_cls_losses, name='neg_cls_loss')
        cls_losses = pos_cls_losses + neg_cls_losses
        self.add_loss(cls_losses)

        return y_reg_pred, y_cls_pred

    def get_config(self):
        config = {
            "n_classes": self.n_classes,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        base_config = super(RetinaLossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
