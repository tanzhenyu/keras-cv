import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from kerascv.data.voc_segmentation import voc_segmentation_dataset_from_directory


BASE_WEIGHT_PATH = ('https://github.com/JonathanCMitchell/mobilenet_v2_keras/'
                    'releases/download/v1.1/')
alpha = 1.0
model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
                          str(alpha) + '_' + str(224) + '_no_top' + '.h5')
weight_path = BASE_WEIGHT_PATH + model_name
num_classes = 21
kernel_reg = tf.keras.regularizers.l2(0.005)


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, name=None, reduction=tf.keras.losses.Reduction.AUTO):
        super(FocalLoss, self).__init__(name=name, reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma

    # y_true [batch_size, H, W]
    # y_pred [batch_size, H, W, n_cls]
    def call(self, y_true, y_pred):
        with tf.name_scope("FocalLoss"):
            alpha_t = tf.cast(self.alpha, tf.float32)
            gamma_t = tf.cast(self.gamma, tf.float32)
            ones = tf.constant(1., dtype=tf.float32)
            y_true = tf.one_hot(y_true, depth=num_classes, on_value=1.0, off_value=0.0)[:, :, :, 1:]
            y_pred = y_pred[:, :, :, 1:]
            x_ent = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
            p = tf.nn.sigmoid(y_pred)
            positive_indices = tf.equal(y_true, ones)
            alpha = tf.where(positive_indices, alpha_t, (ones - alpha_t))
            pt = tf.where(positive_indices, p, ones - p)
            losses = alpha * tf.math.pow(ones - pt, gamma_t) * x_ent
            # [batch_size, H, W]
            return tf.reduce_sum(losses, axis=-1)


class MyIOUMetrics(tf.keras.metrics.MeanIoU):
    def __init__(self, name=None, **kwargs):
        super(MyIOUMetrics, self).__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        super(MyIOUMetrics, self).update_state(y_true, y_pred, sample_weight)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, filters, alpha, stride, expansion, block_id, skip_connection, rate):
    in_channels = inputs.shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding="same",
                          use_bias=False,
                          kernel_regularizer=kernel_reg,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(momentum=0.999, name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               use_bias=False,
                               padding="same",
                               kernel_regularizer=kernel_reg,
                               dilation_rate=(rate, rate),
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding="same",
                      kernel_regularizer=kernel_reg,
                      use_bias=False,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(momentum=0.999, name=prefix + 'project_BN')(x)

    if skip_connection:
        return layers.Add(name=prefix + 'add')([inputs, x])

    return x


def aspp_conv(inp, filters, dilation):
    o = inp
    with tf.name_scope("ASPP_Conv_" + str(dilation)):
        o = layers.Conv2D(filters=filters,
                          kernel_size=(3, 3),
                          padding="same",
                          kernel_regularizer=kernel_reg,
                          dilation_rate=dilation,
                          use_bias=False)(o)
        o = layers.BatchNormalization(momentum=0.999)(o)
        o = layers.ReLU()(o)
    return o


def aspp_pool(inp, filters):
    o = inp
    h, w, c = o.shape[1:]
    with tf.name_scope("ASPP_Pool_" + str(filters)):
        o = layers.GlobalAveragePooling2D()(o)
        o = layers.Reshape((1, 1, c))(o)
        o = layers.Conv2D(filters=filters,
                          kernel_size=(1, 1),
                          kernel_regularizer=kernel_reg,
                          use_bias=False)(o)
        o = layers.BatchNormalization(momentum=0.999)(o)
        o = layers.ReLU()(o)
        # Interpolate
        o = layers.experimental.preprocessing.Resizing(h, w)(o)
    return o


def aspp(inp, dilations):
    out_channels = 256
    results = []
    with tf.name_scope("ASPP"):
        with tf.name_scope("Conv_1x1"):
            x1 = inp
            x1 = layers.Conv2D(filters=out_channels,
                               kernel_size=(1, 1),
                               kernel_regularizer=kernel_reg,
                               use_bias=False)(x1)
            x1 = layers.BatchNormalization(momentum=0.999)(x1)
            x1 = layers.ReLU()(x1)
            results.append(x1)

        for dilation in dilations:
            with tf.name_scope(str(dilation)):
                results.append(aspp_conv(inp, out_channels, dilation))

        results.append(aspp_pool(inp, out_channels))

    o = layers.Concatenate()(results)
    o = layers.Conv2D(filters=out_channels,
                      kernel_size=(1, 1),
                      kernel_regularizer=kernel_reg,
                      use_bias=False)(o)
    o = layers.BatchNormalization(momentum=0.999)(o)
    o = layers.ReLU()(o)
    o = layers.Dropout(0.1)(o)

    return o


def deeplab_head(inp, dilations):
    o = inp
    o = aspp(o, dilations)
    o = layers.Conv2D(256, kernel_size=(3, 3), padding="same", use_bias=False)(o)
    o = layers.BatchNormalization(momentum=0.999)(o)
    o = layers.ReLU()(o)
    o = layers.Conv2D(num_classes, kernel_size=(1, 1))(o)
    return o


def mobilenet_v2(input_shape):
    img_input = layers.Input(shape=input_shape)
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2),
                      padding="same",
                      kernel_regularizer=kernel_reg,
                      use_bias=False,
                      name='Conv1')(img_input)
    x = layers.BatchNormalization(momentum=0.999, name='bn_Conv1')(x)
    x = layers.ReLU(6., name="Conv1_relu")(x)

    # [filters, alpha, stride, expansion, block_id, skip_connection, rate
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1,
                            block_id=0, skip_connection=False, rate=1)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6,
                            block_id=1, skip_connection=False, rate=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6,
                            block_id=2, skip_connection=True, rate=1)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6,
                            block_id=3, skip_connection=False, rate=1)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6,
                            block_id=4, skip_connection=True, rate=1)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6,
                            block_id=5, skip_connection=True, rate=1)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6,
                            block_id=6, skip_connection=False, rate=1)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6,
                            block_id=7, skip_connection=True, rate=2)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6,
                            block_id=8, skip_connection=True, rate=2)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6,
                            block_id=9, skip_connection=True, rate=2)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6,
                            block_id=10, skip_connection=False, rate=2)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6,
                            block_id=11, skip_connection=True, rate=2)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6,
                            block_id=12, skip_connection=True, rate=2)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6,
                            block_id=13, skip_connection=False, rate=2)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6,
                            block_id=14, skip_connection=True, rate=4)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6,
                            block_id=15, skip_connection=True, rate=4)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6,
                            block_id=16, skip_connection=False, rate=4)

    backbone = tf.keras.Model(img_input, x)
    weights_path = tf.keras.utils.get_file(model_name, weight_path, cache_subdir='models')
    backbone.load_weights(weights_path, by_name=True)

    x = deeplab_head(x, dilations=[12, 24, 36])
    x = layers.experimental.preprocessing.Resizing(input_shape[0], input_shape[1])(x)
    model = tf.keras.Model(img_input, x, name='deeplab_v3_mobilenet_v2')
    return model


def train_val_save_deeplab():
    batch_size = 8
    learning_rate = 0.01
    base_size = 513
    crop_size = 513
    epochs = 50
    momentum = 0.9
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    train_voc_ds_2012 = voc_segmentation_dataset_from_directory(
        split="train", batch_size=batch_size, base_size=base_size, crop_size=crop_size,
        preprocess_input=preprocess_input
    )
    eval_voc_ds_2012 = voc_segmentation_dataset_from_directory(
        split="val", batch_size=batch_size, base_size=base_size, crop_size=crop_size,
        preprocess_input=preprocess_input
    )
    # 2913 / 16 * 50
    max_iter = 9000
    lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=learning_rate, end_learning_rate=0.0,
        decay_steps=max_iter, power=0.9
    )

    model = mobilenet_v2([crop_size, crop_size, 3])
    optimizer = tf.keras.optimizers.SGD(lr_scheduler, momentum=momentum)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = FocalLoss()
    iou_metric = MyIOUMetrics()
    top_5_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy()
    model.compile(optimizer, loss, weighted_metrics=["accuracy", iou_metric, top_5_acc])
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="deeplabv3_os8.hdf5", save_best_only=True, monitor="val_my_iou_metrics"
    )
    print('-------------------Start Training DeepLab-------------------')
    print('-------------------Trainable Variables-------------------')
    for var in model.trainable_variables:
        print('var {}, {}'.format(var.name, var.shape))
    model.summary()
    model.fit(train_voc_ds_2012.prefetch(tf.data.experimental.AUTOTUNE), epochs=epochs,
              callbacks=[ckpt_callback], validation_data=eval_voc_ds_2012)


if __name__ == "__main__":
    train_val_save_deeplab()
