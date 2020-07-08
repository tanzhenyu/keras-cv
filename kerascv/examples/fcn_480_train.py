import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from kerascv.data.voc_segmentation import voc_segmentation_dataset_from_directory


num_classes = 21
input_shape = (480, 480, 3)


class MyIOUMetrics(tf.keras.metrics.MeanIoU):
    def __init__(self, name=None, **kwargs):
        super(MyIOUMetrics, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        super(MyIOUMetrics, self).update_state(y_true, y_pred, sample_weight)


def set_upsampling_weight(layer):
    kernel = layer.kernel
    kernel_shape = kernel.shape.as_list()
    kernel_size = kernel_shape[0]
    in_channels = kernel_shape[2]
    out_channels = kernel_shape[3]
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((kernel_size, kernel_size, out_channels, in_channels),
                      dtype=np.float64)
    for i in range(out_channels):
        for j in range(in_channels):
            weight[:, :, i, j] = filt
    kernel.assign(weight)


def get_fcn_32(weights="imagenet"):
    keras_inp = tf.keras.Input(shape=input_shape, name="fcn_32s")
    backbone = tf.keras.applications.vgg16.VGG16(include_top=False, weights=weights, input_tensor=keras_inp)
    x = backbone.outputs[0]
    x = layers.Conv2D(4096, 7, padding="same", activation="relu", name="fc6")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(4096, 1, padding="same", activation="relu", name="fc7")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(num_classes, 1, kernel_initializer="he_normal", name="fcn_32_fc7_pool_conv_1")(x)
    conv_upsample = layers.Conv2DTranspose(num_classes, kernel_size=(64, 64), strides=(32, 32),
                                           use_bias=False, padding="same", name="fcn_32_conv2d_transpose_32")
    x = conv_upsample(x)
    set_upsampling_weight(conv_upsample)
    x = layers.Activation("softmax")(x)
    return tf.keras.Model(keras_inp, x, name="fcn32_vgg16")


def get_fcn_16():
    backbone = tf.keras.models.load_model('fcn_32.hdf5', compile=False)

    o1 = backbone.get_layer("conv2d").output
    conv_upsample = layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False,
                                           padding="same", name="fcn_16_pool5_conv2d_transpose_2")
    o1 = conv_upsample(o1)
    set_upsampling_weight(conv_upsample)

    o2 = backbone.get_layer("block4_pool").output
    o2 = layers.Conv2D(num_classes, 1, kernel_initializer="he_normal", name="fcn_16_block4_pool_conv_1")(o2)

    o = layers.Add()([o1, o2])
    conv_upsample = layers.Conv2DTranspose(num_classes, kernel_size=(32, 32), strides=(16, 16),
                                           use_bias=False, padding="same", name="fcn_16_conv2d_transpose_16")
    o = conv_upsample(o)
    set_upsampling_weight(conv_upsample)
    o = layers.Activation("softmax")(o)
    return tf.keras.Model(backbone.input, o, name="fcn16_vgg16")


def train_val_save_fcn_32():
    batch_size = 20
    train_voc_ds_2012 = voc_segmentation_dataset_from_directory(split="train", batch_size=batch_size)
    eval_voc_ds_2012 = voc_segmentation_dataset_from_directory(split="val", batch_size=batch_size)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        optimizer = tfa.optimizers.SGDW(weight_decay=0.0002, learning_rate=0.001, momentum=0.9)
        model = get_fcn_32()
        iou_metric = MyIOUMetrics()
        model.compile(optimizer, "sparse_categorical_crossentropy", weighted_metrics=["accuracy", iou_metric])
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath='fcn_32.hdf5', save_best_only=True)
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=0.01)

    print('-------------------Start Training FCN32-------------------')
    print('-------------------Trainable Variables-------------------')
    for var in model.trainable_variables:
        print('var {}, {}'.format(var.name, var.shape))
    model.summary()
    # 2913 images is around 150 steps
    model.fit(train_voc_ds_2012.prefetch(tf.data.experimental.AUTOTUNE), epochs=100,
              callbacks=[lr_callback, ckpt_callback], validation_data=eval_voc_ds_2012)


def train_val_save_fcn_16():
    batch_size = 20
    train_voc_ds_2012 = voc_segmentation_dataset_from_directory(split="train", batch_size=batch_size)
    eval_voc_ds_2012 = voc_segmentation_dataset_from_directory(split="val", batch_size=batch_size)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        optimizer = tfa.optimizers.SGDW(weight_decay=0.0002, learning_rate=0.0001, momentum=0.9)
        model = get_fcn_16()
        iou_metric = MyIOUMetrics()
        model.compile(optimizer, "sparse_categorical_crossentropy", weighted_metrics=["accuracy", iou_metric])
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath='fcn_16.hdf5', save_best_only=True)
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=3, min_delta=0.01)

    print('-------------------Start Training FCN16-------------------')
    print('-------------------Trainable Variables-------------------')
    for var in model.trainable_variables:
        print('var {}, {}'.format(var.name, var.shape))
    model.summary()
    # 2913 images is around 150 steps
    model.fit(train_voc_ds_2012.prefetch(tf.data.experimental.AUTOTUNE), epochs=40,
              callbacks=[lr_callback, ckpt_callback], validation_data=eval_voc_ds_2012)


if __name__ == "__main__":
    if not os.path.exists("fcn_32.hdf5"):
        train_val_save_fcn_32()
    elif not os.path.exists("fcn_16.hdf5"):
        train_val_save_fcn_16()
