import os
import tensorflow as tf
from tensorflow.keras import layers
from kerascv.data.voc_segmentation import voc_segmentation_dataset_from_directory


def get_fcn_32(weights_path, input_shape, n_classes=21):
    keras_inp = tf.keras.Input(shape=input_shape, name="fcn_32s")
    backbone = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_tensor=keras_inp)
    x = backbone.outputs[0]
    x = layers.Conv2D(4096, 7, padding="same", activation="relu", name="fc6")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(4096, 1, padding="same", activation="relu", name="fc7")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(n_classes, 1, kernel_initializer="he_normal")(x)
    conv_upsample = layers.Conv2DTranspose(n_classes, kernel_size=(64, 64), strides=(32, 32), use_bias=False,
                                           padding="same")
    x = conv_upsample(x)
    x = layers.Activation("softmax")(x)
    model = tf.keras.Model(keras_inp, x, name="fcn32_vgg16")
    model.load_weights(weights_path)
    return model


def eval_fcn_32():
    batch_size = 20
    weights_path = os.path.join(os.getcwd(), 'fcn_32.hdf5')
    eval_voc_ds_2012 = voc_segmentation_dataset_from_directory(split="val", batch_size=batch_size)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_shape = (480, 480, 3)
        iou_metric = tf.keras.metrics.MeanIoU(num_classes=21)
        model = get_fcn_32(weights_path, input_shape)
        y_pred = model.outputs[0]
        y_pred = tf.math.argmax(y_pred, axis=-1)
        inputs = model.inputs
        eval_model = tf.keras.Model(inputs, y_pred)
        eval_model.compile([iou_metric])
    print('-------------------Start Evaluating-------------------')
    eval_model.evaluate(eval_voc_ds_2012)


if __name__ == "__main__":
    eval_fcn_32()
