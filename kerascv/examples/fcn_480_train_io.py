import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import load_img
from kerascv.examples.fcn_480_train import MyIOUMetrics, get_fcn_32


batch_size = 20
input_shape = (480, 480, 3)
img_size = input_shape[:2]
num_classes = 21
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

    def __init__(self, input_img_paths, target_img_paths):
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


def train_eval_io():
    train_gen = VOCData(train_input_img_paths, train_target_img_paths)
    val_gen = VOCData(eval_input_img_paths, eval_target_img_paths)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_fcn_32()
        optimizer = tfa.optimizers.SGDW(weight_decay=0.0002, learning_rate=0.001, momentum=0.9)
        iou_metric = MyIOUMetrics(num_classes)
        model.compile(optimizer, "sparse_categorical_crossentropy", ["accuracy", iou_metric])
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath='fcn_32_io.hdf5', save_best_only=True)
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=0.01)

    print('-------------------Start Training FCN32 IO-------------------')
    print('-------------------Trainable Variables-------------------')
    for var in model.trainable_variables:
        print('var {}, {}'.format(var.name, var.shape))
    model.summary()
    # 2913 images is around 150 steps
    model.fit(train_gen, epochs=100,
              callbacks=[lr_callback, ckpt_callback], validation_data=val_gen)


if __name__ == "__main__":
    train_eval_io()
