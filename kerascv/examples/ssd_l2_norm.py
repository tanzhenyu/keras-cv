import tensorflow as tf


class L2Normalization(tf.keras.layers.Layer):
    """L2 Normalization from ParseNet, with a trainable scaling parameter

    Reference:
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    """

    def __init__(self, gamma=20, name=None, **kwargs):
        self.gamma = gamma
        self.channel_axis = -1
        super(L2Normalization, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        channels = tf.compat.dimension_value(input_shape[self.channel_axis])
        gamma_initializer = tf.keras.initializers.Constant(self.gamma)
        self.gamma = self.add_weight(
            name="gamma",
            shape=(channels,),
            dtype=tf.float32,
            initializer=gamma_initializer,
        )
        super(L2Normalization, self).build(input_shape)

    def call(self, x):
        output = tf.math.l2_normalize(x, self.channel_axis)
        return output * tf.cast(self.gamma, x.dtype)

    def get_config(self):
        config = {"gamma": self.gamma}
        base_config = super(L2Normalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
