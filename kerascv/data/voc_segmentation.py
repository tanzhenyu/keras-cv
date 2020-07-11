import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter


def voc_segmentation_dataset_from_directory(
        directory=None,
        base_size=520,
        crop_size=480,
        batch_size=20,
        split="train",
        shuffle=True,
        seed=None,
        preprocess_input=tf.keras.applications.vgg16.preprocess_input,
        n_classes=21,
):
    directory = directory or os.path.expanduser('~/VOCdevkit/VOC2012')
    if not os.path.isdir(directory):
        raise ValueError("Directory Not Found {}".format(directory))
    mask_dir = os.path.join(directory, "SegmentationClass")
    image_dir = os.path.join(directory, "JPEGImages")
    splits_dir = os.path.join(directory, "ImageSets/Segmentation")
    if split == "train":
        splits_dir = os.path.join(splits_dir, "trainval.txt")
    elif split == "val":
        splits_dir = os.path.join(splits_dir, "val.txt")
    elif split == "test":
        splits_dir = os.path.join(splits_dir, "test.txt")
    else:
        raise ValueError("Unknown split {}".format(split))

    random.seed(seed)

    def file_generator():
        with tf.io.gfile.GFile(splits_dir, mode="r") as f:
            lines = f.readlines()
            for line in lines:
                image_file = os.path.join(image_dir, line.rstrip("\n") + ".jpg")
                mask_file = os.path.join(mask_dir, line.rstrip("\n") + ".png")
                img_pil = Image.open(image_file)
                mask_pil = Image.open(mask_file)
                if random.random() < 0.5:
                    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
                long_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
                w, h = img_pil.size
                if h > w:
                    oh = long_size
                    ow = int(1.0 * w * long_size / h + 0.5)
                    short_size = ow
                else:
                    ow = long_size
                    oh = int(1.0 * h * long_size / w + 0.5)
                    short_size = oh
                img_pil = img_pil.resize((ow, oh), Image.BILINEAR)
                mask_pil = mask_pil.resize((ow, oh), Image.NEAREST)
                # pad crop
                if short_size < crop_size:
                    padh = crop_size - oh if oh < crop_size else 0
                    padw = crop_size - ow if ow < crop_size else 0
                    img_pil = ImageOps.expand(img_pil, border=(0, 0, padw, padh), fill=0)
                    mask_pil = ImageOps.expand(mask_pil, border=(0, 0, padw, padh), fill=255)
                # random crop crop_size
                w, h = img_pil.size
                x1 = random.randint(0, w - crop_size)
                y1 = random.randint(0, h - crop_size)
                img_pil = img_pil.crop((x1, y1, x1 + crop_size, y1 + crop_size))
                mask_pil = mask_pil.crop((x1, y1, x1 + crop_size, y1 + crop_size))
                # gaussian blur as in PSP
                if random.random() < 0.5:
                    img_pil = img_pil.filter(ImageFilter.GaussianBlur(
                        radius=random.random()))
                # preprocess image before returning
                img = np.array(img_pil)
                if preprocess_input is not None:
                    img = preprocess_input(img)
                mask = np.array(mask_pil)
                sample_weights = np.ones_like(mask, dtype=np.float)
                ignore_mask_indices = (mask == 255)
                sample_weights[ignore_mask_indices] = 0.
                mask[ignore_mask_indices] = 0
                # Automatically convert palette mode to grayscale with class index.
                yield img, mask, sample_weights

    img_ds = tf.data.Dataset.from_generator(file_generator, (tf.float32, tf.uint8, tf.float32))

    def set_shape_fn(img, mask, sample_weights):
        img.set_shape([crop_size, crop_size, 3])
        mask.set_shape([crop_size, crop_size])
        # mask_one_hot = tf.one_hot(mask, depth=n_classes)
        # mask_one_hot.set_shape([crop_size, crop_size, n_classes])
        sample_weights.set_shape([crop_size, crop_size])
        # return img, mask_one_hot, sample_weights
        return img, mask, sample_weights

    if shuffle:
        img_ds = img_ds.shuffle(buffer_size=8 * batch_size, seed=seed)
    img_ds = img_ds.map(set_shape_fn)
    img_ds = img_ds.batch(batch_size)
    return img_ds
