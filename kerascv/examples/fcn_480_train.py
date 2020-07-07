import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from kerascv.data.voc_segmentation import voc_segmentation_dataset_from_directory


def set_upsampling_weight(layer):
    kernel = layer.kernel
    kernel_shape = kernel.shape.as_list()
    print('kernel shape {}'.format(kernel_shape))
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


def get_fcn_32(input_shape, n_classes=21):
    keras_inp = tf.keras.Input(shape=input_shape, name="fcn_32s")
    backbone = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_tensor=keras_inp)
    x = backbone.outputs[0]
    x = layers.Conv2D(4096, 7, padding="same", activation="relu", name="fc6")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(4096, 1, padding="same", activation="relu", name="fc7")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(n_classes, 1, kernel_initializer="he_normal")(x)
    conv_upsample = layers.Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(32, 32), use_bias=False, padding="same")
    x = conv_upsample(x)
    set_upsampling_weight(conv_upsample)
    x = layers.Activation("softmax")(x)
    return tf.keras.Model(keras_inp, x, name="fcn32_vgg16")


def train_val_save_fcn_32():
    batch_size = 20
    train_voc_ds_2012 = voc_segmentation_dataset_from_directory(split="train", batch_size=batch_size)
    eval_voc_ds_2012 = voc_segmentation_dataset_from_directory(split="val", batch_size=batch_size)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_shape = (480, 480, 3)
        loss = tf.keras.losses.CategoricalCrossentropy()
        acc_metric = tf.keras.metrics.CategoricalAccuracy()
        loss_metric = tf.keras.metrics.CategoricalCrossentropy()
        pr_metric = tf.keras.metrics.Precision()
        re_metric = tf.keras.metrics.Recall()
        optimizer = tfa.optimizers.SGDW(weight_decay=0.0002, learning_rate=0.001, momentum=0.9)
        model = get_fcn_32(input_shape)
        model.compile(optimizer, loss, [acc_metric, loss_metric, pr_metric, re_metric])
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='./fcn_32_weights/fcn32.{epoch:02d}-{val_loss:.2f}.hdf5',
            save_weights_only=True, save_best_only=True)

    print('-------------------Start Training-------------------')
    print('-------------------Trainable Variables-------------------')
    for var in model.trainable_variables:
        print('var {}, {}'.format(var.name, var.shape))
    # 2913 images is around 150 steps
    model.fit(train_voc_ds_2012.prefetch(tf.data.experimental.AUTOTUNE), epochs=10,
              callbacks=[ckpt_callback], validation_data=eval_voc_ds_2012)


if __name__ == "__main__":
    train_val_save_fcn_32()
