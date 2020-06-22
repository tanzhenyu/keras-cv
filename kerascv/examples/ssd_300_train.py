from kerascv.layers.anchor_generators.multi_scale_anchor_generator import MultiScaleAnchorGenerator
from kerascv.layers.iou_similarity import IOUSimilarity
from kerascv.layers.losses.ssd_loss_layer import SSDLossLayer
from kerascv.layers.matchers.greedy_bipartite import target_assign_tf_func
from kerascv.layers.ssd_box_coder import SSDBoxCoder
from kerascv.examples.ssd_l2_norm import L2Normalization

import numpy as np
import tensorflow as tf
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

WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


# The reason you cannot leave VGG16 backbone as a separate function is that you need conv4_3!
def build_ssd_vgg16_fpn(input_shape, l2_reg=0.0005):
    feature_maps = []
    inputs = layers.Input(shape=input_shape)
    x = inputs

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_1')(x)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
    pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
    pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
    pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3')(conv4_2)
    pool4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_3')(conv5_2)
    pool5 = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

    vgg_model = tf.keras.Model(inputs, pool5, name='vgg16')
    weights_path = tf.keras.utils.get_file(
        'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        file_hash='6d6bbae143d832006294945121d1f1fc')
    vgg_model.load_weights(weights_path)

    conv4_3_norm = L2Normalization()(conv4_3)
    feature_maps.append(conv4_3_norm)
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc6')(pool5)

    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7')(fc6)
    feature_maps.append(fc7)

    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_1')(fc7)
    conv6_1_pad = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1_pad)
    feature_maps.append(conv6_2)

    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_1')(conv6_2)
    conv7_1_pad = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1_pad)
    feature_maps.append(conv7_2)

    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)
    feature_maps.append(conv8_2)

    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)
    feature_maps.append(conv9_2)

    model = tf.keras.Model(inputs=inputs, outputs=feature_maps)

    return model


def build_ssd_vgg16_head(ssd_vgg16_fpn, l2_reg=0.0005, n_classes=21):
    feature_maps = ssd_vgg16_fpn.outputs
    assert len(feature_maps) == 6
    conv4_3, fc7, conv6_2, conv7_2, conv8_2, conv9_2 = feature_maps
    n_boxes = [4, 6, 6, 6, 4, 4]

    conv4_3_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_loc')(conv4_3)
    fc7_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_loc')(fc7)
    conv6_2_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_loc')(conv6_2)
    conv7_2_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_loc')(conv7_2)
    conv8_2_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_loc')(conv8_2)
    conv9_2_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_loc')(conv9_2)

    conv4_3_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_conf')(conv4_3)
    fc7_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_conf')(fc7)
    conv6_2_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_conf')(conv6_2)
    conv7_2_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_conf')(conv7_2)
    conv8_2_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_conf')(conv8_2)
    conv9_2_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_conf')(conv9_2)

    conv4_3_loc_reshape = Reshape((-1, 4), name='conv4_3_loc_reshape')(conv4_3_loc)
    fc7_loc_reshape = Reshape((-1, 4), name='fc7_loc_reshape')(fc7_loc)
    conv6_2_loc_reshape = Reshape((-1, 4), name='conv6_2_loc_reshape')(conv6_2_loc)
    conv7_2_loc_reshape = Reshape((-1, 4), name='conv7_2_loc_reshape')(conv7_2_loc)
    conv8_2_loc_reshape = Reshape((-1, 4), name='conv8_2_loc_reshape')(conv8_2_loc)
    conv9_2_loc_reshape = Reshape((-1, 4), name='conv9_2_loc_reshape')(conv9_2_loc)

    conv4_3_conf_reshape = Reshape((-1, n_classes), name='conv4_3_conf_reshape')(conv4_3_conf)
    fc7_conf_reshape = Reshape((-1, n_classes), name='fc7_conf_reshape')(fc7_conf)
    conv6_2_conf_reshape = Reshape((-1, n_classes), name='conv6_2_conf_reshape')(conv6_2_conf)
    conv7_2_conf_reshape = Reshape((-1, n_classes), name='conv7_2_conf_reshape')(conv7_2_conf)
    conv8_2_conf_reshape = Reshape((-1, n_classes), name='conv8_2_conf_reshape')(conv8_2_conf)
    conv9_2_conf_reshape = Reshape((-1, n_classes), name='conv9_2_conf_reshape')(conv9_2_conf)

    bbox_loc = Concatenate(axis=1, name='bbox_loc_pred')([conv4_3_loc_reshape, fc7_loc_reshape, conv6_2_loc_reshape, conv7_2_loc_reshape, conv8_2_loc_reshape, conv9_2_loc_reshape])
    bbox_conf = Concatenate(axis=1, name='bbox_conf_pred')([conv4_3_conf_reshape, fc7_conf_reshape, conv6_2_conf_reshape, conv7_2_conf_reshape, conv8_2_conf_reshape, conv9_2_conf_reshape])
    bbox_softmax = tf.keras.layers.Activation('softmax', name='bbox_softmax')(bbox_conf)
    return bbox_loc, bbox_softmax


image_size = [300, 300]
# The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]

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
anchor_generator = MultiScaleAnchorGenerator(image_size=image_size, scales=ssd_vgg16_scales, aspect_ratios=ssd_vgg16_aspect_ratios)
anchors = anchor_generator(feature_map_sizes)
similarity_cal = IOUSimilarity()
box_encoder = SSDBoxCoder(center_variances=[.1, .1], size_variances=[.2, .2])
box_decoder = SSDBoxCoder(center_variances=[.1, .1], size_variances=[.2, .2], invert=True)
mean_color = np.asarray([123, 117, 104])
ssd_loss_layer = SSDLossLayer()

# Add background as class 0
voc_classes = ['background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']


def encode_flatten_map(features):
    # image in the range of [0, 255] before subtract
    image = tf.cast(features['image'] - mean_color, tf.float32)
    image = tf.image.resize(image, [300, 300])
    # normalized corner format
    gt_boxes = features['objects']['bbox']
    gt_labels = features['objects']['label']
    # reserve 0 for background label
    gt_labels = gt_labels + 1
    # expand dimension for future encoding
    gt_labels = gt_labels[:, tf.newaxis]
    return image, gt_boxes, gt_labels


def assigned_gt_fn(image, gt_boxes, gt_labels):
    matched_gt_boxes, matched_gt_labels, positive_mask, negative_mask = target_assign_tf_func(gt_boxes, gt_labels, anchors)
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


def main():
    voc_ds_2007 = tfds.load('voc/2007')
    voc_ds_2012 = tfds.load('voc/2012')

    train_voc_ds = voc_ds_2007['train'].concatenate(voc_ds_2012['train'])
    train_voc_ds = train_voc_ds.shuffle(buffer_size=100)
    encoded_voc_train_ds = train_voc_ds.map(encode_flatten_map).map(assigned_gt_fn)

    ssd_vgg16_fpn = build_ssd_vgg16_fpn((300, 300, 3), l2_reg=0.005)
    gt_loc_pred, gt_cls_pred = build_ssd_vgg16_head(ssd_vgg16_fpn, l2_reg=0.005)
    gt_loc_input = Input((8732, 4), dtype=tf.float32, name='gt_loc_true')
    gt_cls_input = Input((8732,), dtype=tf.int64, name='gt_cls_true')
    positive_mask = Input((8732,), dtype=tf.float32, name='positive_mask')
    negative_mask = Input((8732,), dtype=tf.float32, name='negative_mask')
    gt_final_loc_pred, gt_final_cls_pred = ssd_loss_layer(gt_loc_input, gt_loc_pred, gt_cls_input, gt_cls_pred,
                                                          positive_mask, negative_mask)

    model_inputs = {'image': ssd_vgg16_fpn.inputs[0],
                    'matched_gt_boxes': gt_loc_input,
                    'matched_gt_labels': gt_cls_input,
                    'positive_mask': positive_mask,
                    'negative_mask': negative_mask}
    model_outputs = [gt_final_loc_pred, gt_final_cls_pred]
    train_model = Model(inputs=model_inputs, outputs=model_outputs)

    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(
        schedule=lr_scheduler, verbose=1)
    optimizer = tf.keras.optimizers.Adam()
    train_model.compile(optimizer)

    print('-------------------Start Training-------------------')
    train_model.fit(encoded_voc_train_ds.batch(32).prefetch(1000).cache(), epochs=400,
                    callbacks=[learning_rate_scheduler])

    print('-------------------Start Evaluating-------------------')
    test_voc_ds = voc_ds_2007['test'].concatenate(voc_ds_2012['test'])
    test_voc_ds = test_voc_ds.shuffle(buffer_size=100)
    encoded_voc_test_ds = test_voc_ds.map(encode_flatten_map).map(assigned_gt_fn)
    train_model.evaluate(encoded_voc_test_ds.batch(32))

    print('-------------------Start Saving-------------------')
    train_model.save('ssd_300.h5')


if __name__ == "main":
    main()