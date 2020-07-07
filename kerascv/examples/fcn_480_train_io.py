import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers


batch_size = 20
input_shape = (480, 480, 3)
img_size = input_shape[:2]
directory = os.path.expanduser('~/VOCdevkit/VOC2012')
mask_dir = os.path.join(directory, "SegmentationClass")
image_dir = os.path.join(directory, "JPEGImages")

train_splits_dir = os.path.join(directory, "ImageSets/Segmentation/trainval.txt")
train_file_ids = []
with tf.io.gfile.GFile(train_splits_dir, mode="r") as f:
    lines = f.readlines()
    for line in lines:
        train_file_ids.append(line)
train_input_img_paths = sorted(
    [
        os.path.join(image_dir, file_id.rstrip("\n") + ".jpg")
        for file_id in train_file_ids
    ]
)
train_target_img_paths = sorted(
    [
        os.path.join(mask_dir, file_id.rstrip("\n") + ".png")
        for file_id in train_file_ids
    ]
)

print("Number of samples:", len(train_input_img_paths))

for input_path, target_path in zip(train_input_img_paths[:10], train_target_img_paths[:10]):
    print(input_path, "|", target_path)


eval_splits_dir = os.path.join(directory, "ImageSets/Segmentation/val.txt")
eval_file_ids = []
with tf.io.gfile.GFile(eval_splits_dir, mode="r") as f:
    lines = f.readlines()
    for line in lines:
        eval_file_ids.append(line)
eval_input_img_paths = sorted(
    [
        os.path.join(image_dir, file_id.rstrip("\n") + ".jpg")
        for file_id in eval_file_ids
    ]
)
eval_target_img_paths = sorted(
    [
        os.path.join(mask_dir, file_id.rstrip("\n") + ".png")
        for file_id in eval_file_ids
    ]
)


class VOCData(tf.keras.utils.Sequence):

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
        return x, y


def get_fcn_32(n_classes=21):
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
    x = layers.Activation("softmax")(x)
    return tf.keras.Model(keras_inp, x, name="fcn32_vgg16")


def train_eval_io():
    train_gen = VOCData(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    val_gen = VOCData(batch_size, img_size, eval_input_img_paths, eval_target_img_paths)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_fcn_32()
        optimizer = tfa.optimizers.SGDW(weight_decay=0.0002, learning_rate=0.001, momentum=0.9)
        acc_metric = tf.keras.metrics.CategoricalAccuracy()
        pr_metric = tf.keras.metrics.Precision()
        re_metric = tf.keras.metrics.Recall()
        model.compile(optimizer, "sparse_categorical_crossentropy", [acc_metric, pr_metric, re_metric])
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='./fcn_32_weights_io/fcn32.{epoch:02d}-{val_loss:.2f}.hdf5',
            save_best_only=True)

    print('-------------------Start Training-------------------')
    print('-------------------Trainable Variables-------------------')
    for var in model.trainable_variables:
        print('var {}, {}'.format(var.name, var.shape))
    # 2913 images is around 150 steps
    model.fit(train_gen, epochs=10, callbacks=[ckpt_callback], validation_data=val_gen)


if __name__ == "__main__":
    train_eval_io()
