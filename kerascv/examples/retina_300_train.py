from kerascv.layers.anchor_generators.multi_scale_anchor_generator import MultiScaleAnchorGenerator
from kerascv.layers.iou_similarity import IOUSimilarity
from kerascv.layers.losses.retina_loss_layer import RetinaLossLayer
from kerascv.layers.matchers.argmax_matcher import target_assign_argmax
from kerascv.layers.ssd_box_coder import SSDBoxCoder
from kerascv.examples.preprocessing.color_transforms import *
from kerascv.examples.preprocessing.geometric_transforms import *

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

layers = tf.keras.layers
l2 = tf.keras.regularizers.L2
Conv2D = tf.keras.layers.Conv2D
MaxPool2D = tf.keras.layers.MaxPool2D
ZeroPadding2D = tf.keras.layers.ZeroPadding2D
Reshape = tf.keras.layers.Reshape
Concatenate = tf.keras.layers.Concatenate
Input = tf.keras.Input
Model = tf.keras.Model


def build_retina_resnet_fpn():
    inputs = layers.Input(shape=[None, None, 3])
    backbone = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    up_sampler = layers.UpSampling2D(2)
    C3, C4, C5 = [
        backbone.get_layer(l_name).output for l_name in [
            "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"
        ]
    ]
    feature_maps = []
    P3 = Conv2D(256, 1, 1, "same", name="fpn_conv_p3_1")(C3)
    P4 = Conv2D(256, 1, 1, "same", name="fpn_conv_p4_1")(C4)
    P5 = Conv2D(256, 1, 1, "same", name="fpn_conv_p5_1")(C5)
    # The upsampled map is merged with the bottom-up map, which undergoes a 1x1 conv layer to reduce
    # channel dimensions by element-wise addition.
    P4 = P4 + up_sampler(P5)
    P3 = P3 + up_sampler(P4)
    # Append a 3x3 conv on each merged map to generate the final feature map, which is to reduce
    # the aliasing effect of upsampling.
    P3 = Conv2D(256, 3, 1, "same", name="fpn_conv_p3_3")(P3)
    P4 = Conv2D(256, 3, 1, "same", name="fpn_conv_p4_3")(P4)
    P5 = Conv2D(256, 3, 1, "same", name="fpn_conv_p5_3")(P5)
    # Don't use the high-resolution pyramid level P2
    # for computational reasons.
    feature_maps.append(P3)
    feature_maps.append(P4)
    feature_maps.append(P5)
    # P6 is obtained via a 3x3 stride-2 conv on C5
    P6 = Conv2D(256, 3, 2, "same", name="fpn_conv_p6_3")(C5)
    feature_maps.append(P6)
    # P7 is computed by applying Relu followed by a 3x3 stride-2 conv on P6.
    P7 = Conv2D(256, 3, 2, "same", name="fpn_conv_p7_3")(tf.nn.relu(P6))
    feature_maps.append(P7)
    # The final set of feature maps is {P3, P4, P5, P6, P7}.
    model = Model(inputs=inputs, outputs=feature_maps)
    return model


class RetinaNetHead(layers.Layer):
    def __init__(self, num_anchors, output_size, bias_init, name="RetinaHead", **kwargs):
        # All new conv layers are initialized with a Gaussian weight with sigma = 0.01.
        kernel_init = tf.keras.initializers.RandomNormal(0.0, 0.01)
        self.bias_init = bias_init
        self.num_anchors = num_anchors
        self.output_size = output_size
        super(RetinaNetHead, self).__init__(name=name, **kwargs)
        # Parameters of head are shared across all pyramid levels. Take an input feature map with C channels,
        # the head applies four 3x3 conv layers, each with C filters and followed by Relu, finally followed
        # by a 3x3 conv layer with K*A or 4A filters. Use C=256 and A=9 in ResNet50.
        heads = []
        for _ in range(4):
            # All new conv layers except the final one are initialized with bias b = 0
            heads.append(
                Conv2D(256, 3, padding="same", kernel_initializer=kernel_init, activation='relu')
            )
        heads.append(
            Conv2D(num_anchors * output_size, 3, 1, padding="same", kernel_initializer=kernel_init,
                   bias_initializer=bias_init)
        )
        self.heads = heads

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        output = inputs
        for head in self.heads:
            output = head(output)
        return tf.reshape(output, [batch_size, height * width, self.output_size])

    def get_config(self):
        config = {
            "num_anchors": self.num_anchors,
            "output_size": self.output_size,
            "bias_init": tf.keras.initializers.serialize(self.bias_init),
        }
        base_config = super(RetinaNetHead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RetinaNet(Model):
    def __init__(self, n_classes=80, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.fpn = build_retina_resnet_fpn()
        self.n_classes = n_classes
        # The final conv layer of the cls subnet, b = -log(1- pi) / pi), where pi specifies that at start
        # of training every anchor should be labeled as foreground with confidence of pi.
        prior_init = tf.keras.initializers.Constant(-np.log((1 - 0.01) / 0.01))
        self.box_heads = RetinaNetHead(9 * 4, "zeros")
        self.cls_heads = RetinaNetHead(9 * n_classes, prior_init)

    def call(self, image, training=True):
        feature_maps = self.fpn(image, training=training)
        box_preds = []
        cls_preds = []
        for feature_map in feature_maps:
            box_preds.append(self.box_heads(feature_map))
            cls_preds.append(self.cls_heads(feature_map))
        box_pred = tf.concat(box_preds, axis=1)
        cls_pred = tf.concat(cls_preds, axis=1)
        return box_pred, cls_pred


# At each pyramid level we use anchors at 3 aspect ratios {1:2, 1:1, 2:1}
# For denser scale coverage than FPN, at each level add anchors of sizes {2^0, 2^1/3, 2^2/3}
# The anchor box scales used in the original Retina ResNet50 for the COCO dataset.
scales = [32, 64, 128, 256, 512]

ssd_vgg16_scales = [
                    [scales[0], scales[0], scales[0], np.sqrt(scales[0] * scales[1])],
                    [scales[1], scales[1], scales[1], scales[1], scales[1], np.sqrt(scales[1] * scales[2])],
                    [scales[2], scales[2], scales[2], scales[2], scales[2], np.sqrt(scales[2] * scales[3])],
                    [scales[3], scales[3], scales[3], scales[3], scales[3], np.sqrt(scales[3] * scales[4])],
                    [scales[4], scales[4], scales[4], np.sqrt(scales[4] * scales[5])],
                    [scales[5], scales[5], scales[5], np.sqrt(scales[5] * scales[6])],
                    ]
ssd_vgg16_aspect_ratios = [
                           [1., 2., 1/2, 1.],
                           [1., 2., 3., 1/2, 1/3, 1.],
                           [1., 2., 3., 1/2, 1/3, 1.],
                           [1., 2., 3., 1/2, 1/3, 1.],
                           [1., 2., 1/2, 1.],
                           [1., 2., 1/2, 1.]
                           ]
feature_map_sizes = [[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]]
anchor_generator = MultiScaleAnchorGenerator(scales=ssd_vgg16_scales, aspect_ratios=ssd_vgg16_aspect_ratios)
anchors = anchor_generator(image_size, feature_map_sizes)
similarity_cal = IOUSimilarity()
box_encoder = SSDBoxCoder(center_variances=[.1, .1], size_variances=[.2, .2])
box_decoder = SSDBoxCoder(center_variances=[.1, .1], size_variances=[.2, .2], invert=True)
retina_loss_layer = RetinaLossLayer(n_classes=21)

# Add background as class 0
voc_classes = ['background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']


def flatten_and_preprocess(features):
    # image in the range of [0, 255], tf.uint8
    image = features['image']
    # [num_gt_boxes, 4] normalized corner format
    gt_boxes = features['objects']['bbox']
    # [num_gt_boxes]
    gt_labels = features['objects']['label']
    image = photometric_transform(image)
    image, gt_boxes = random_flip_horizontal(image, gt_boxes, normalized=True)
    image = tf.keras.applications.resnet.preprocess_input(image)
    image = tf.image.resize(image, [300, 300])
    # reserve 0 for background label
    gt_labels = gt_labels + 1
    # expand dimension for future encoding
    gt_labels = gt_labels[:, tf.newaxis]
    return image, gt_boxes, gt_labels


def assigned_gt_fn(image, gt_boxes, gt_labels):
    matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask = target_assign_argmax(gt_boxes, gt_labels, anchors, negative_iou_threshold=0.4)
    encoded_matched_gt_boxes = box_encoder(matched_gt_boxes, anchors)
    matched_gt_labels = tf.squeeze(matched_gt_labels, axis=-1)
    return {'image': image, 'matched_gt_boxes': encoded_matched_gt_boxes, 'matched_gt_labels': matched_gt_labels,
            'positive_mask': positive_mask, 'negative_mask': negative_mask}


def lr_scheduler(epoch, lr):
    # decay learning rate at epoch 80 and 100
    if epoch == 80 or epoch == 100:
        return 0.1 * lr
    else:
        return lr


def train_eval_save():
    voc_ds_2007 = tfds.load('voc/2007')
    voc_ds_2012 = tfds.load('voc/2012')

    train_voc_ds = voc_ds_2007['train'].concatenate(voc_ds_2007['validation']).concatenate(
        voc_ds_2012['train']).concatenate(voc_ds_2012['validation'])
    train_voc_ds = train_voc_ds.shuffle(buffer_size=100)
    encoded_voc_train_ds = train_voc_ds.map(flatten_and_preprocess).map(assigned_gt_fn).batch(32)

    test_voc_ds = voc_ds_2007['validation'].concatenate(voc_ds_2012['validation'])
    test_voc_ds = test_voc_ds.shuffle(buffer_size=100)
    encoded_voc_test_ds = test_voc_ds.map(flatten_and_preprocess).map(assigned_gt_fn).batch(32).take(10)

    retina_vgg16_fpn = build_retina_vgg16_fpn((300, 300, 3))
    gt_loc_pred, gt_cls_pred = build_retina_vgg16_head(retina_vgg16_fpn)
    gt_loc_input = Input((8732, 4), dtype=tf.float32, name='gt_loc_true')
    gt_cls_input = Input((8732,), dtype=tf.int64, name='gt_cls_true')
    positive_mask = Input((8732,), dtype=tf.float32, name='positive_mask')
    negative_mask = Input((8732,), dtype=tf.float32, name='negative_mask')
    gt_final_loc_pred, gt_final_cls_pred = retina_loss_layer(gt_loc_input, gt_loc_pred, gt_cls_input, gt_cls_pred,
                                                             positive_mask, negative_mask)

    model_inputs = {'image': retina_vgg16_fpn.inputs[0],
                    'matched_gt_boxes': gt_loc_input,
                    'matched_gt_labels': gt_cls_input,
                    'positive_mask': positive_mask,
                    'negative_mask': negative_mask}
    model_outputs = [gt_final_loc_pred, gt_final_cls_pred]
    train_model = Model(inputs=model_inputs, outputs=model_outputs)

    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(
        schedule=lr_scheduler, verbose=1)
    optimizer = tfa.optimizers.AdamW(weight_decay=0.0005, learning_rate=0.001)
    train_model.compile(optimizer)
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./retina_weights/my_retina.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                       save_weights_only=True, save_best_only=True)

    print('-------------------Start Training-------------------')
    train_model.fit(encoded_voc_train_ds.prefetch(1000), epochs=350,
                    callbacks=[learning_rate_scheduler, ckpt_callback], validation_data=encoded_voc_test_ds)


if __name__ == "__main__":
    train_eval_save()