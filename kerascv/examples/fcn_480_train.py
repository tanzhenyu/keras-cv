import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from kerascv.data.voc_segmentation import voc_segmentation_dataset_from_directory


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
    weight = np.zeros((kernel_size, kernel_size, in_channels, out_channels),
                      dtype=np.float64)
    for i in range(kernel_size):
        for j in range(kernel_size):
            weight[i, j, :, :] = filt
    kernel.assign(weight)


def get_fcn_32(input_shape, n_classes=21):
    keras_inp = tf.keras.Input(shape=input_shape, name="fcn_32s")
    backbone = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_tensor=keras_inp)
    x = backbone.outputs[0]
    x = layers.Conv2D(4096, 7, padding="same", activation="relu", name="fc6")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(4096, 1, padding="same", activation="relu", name="fc7")(x)
    x = layers.Dropout(0.5)(x)
    conv_upsample = layers.Conv2D(n_classes, 1, kernel_initializer="he_normal")
    x = conv_upsample(x)
    set_upsampling_weight(conv_upsample)
    x = layers.Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(32, 32), use_bias=False, padding="same")(x)
    x = layers.Activation("softmax")(x)
    return tf.keras.Model(keras_inp, x, name="fcn32_vgg16")


def train_val_save_fcn_32():
    train_voc_ds_2012 = voc_segmentation_dataset_from_directory(split="train")
    eval_voc_ds_2012 = voc_segmentation_dataset_from_directory(split="val")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_shape = (480, 480, 3)
        loss = tf.keras.losses.CategoricalCrossentropy()
        acc_metric = tf.keras.metrics.CategoricalAccuracy()
        loss_metric = tf.keras.metrics.CategoricalCrossentropy()
        # iou_metric = tf.keras.metrics.MeanIoU(num_classes=21)
        tp_metric = tf.keras.metrics.TruePositives()
        fp_metric = tf.keras.metrics.FalsePositives()
        fn_metric = tf.keras.metrics.FalseNegatives()
        optimizer = tfa.optimizers.SGDW(weight_decay=0.0002, learning_rate=0.001, momentum=0.9)
        model = get_fcn_32(input_shape)
        model.compile(optimizer, loss, [acc_metric, loss_metric, tp_metric, fp_metric, fn_metric])
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='./fcn_32_weights/fcn32.{epoch:02d}-{val_loss:.2f}.hdf5',
            save_weights_only=True, save_best_only=True)

    print('-------------------Start Training-------------------')
    # 2913 images is around 150 steps
    model.fit(train_voc_ds_2012.prefetch(tf.data.experimental.AUTOTUNE), epochs=10,
              callbacks=[ckpt_callback], validation_data=eval_voc_ds_2012)


if __name__ == "__main__":
    train_val_save_fcn_32()