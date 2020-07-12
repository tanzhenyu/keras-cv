import os
import tensorflow as tf
from kerascv.data.voc_segmentation import voc_segmentation_dataset_from_directory


class MyIOUMetrics(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name=None, **kwargs):
        super(MyIOUMetrics, self).__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        super(MyIOUMetrics, self).update_state(y_true, y_pred, sample_weight)


def eval_deeplab(weights_path):
    batch_size = 8
    base_size = 513
    crop_size = 513
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    if os.path.exists(weights_path):
        eval_voc_ds_2012 = voc_segmentation_dataset_from_directory(
            split="val", batch_size=batch_size, base_size=base_size, crop_size=crop_size,
            preprocess_input=preprocess_input
        )
        iou_metric = MyIOUMetrics(num_classes=21)
        top_k_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy()
        model = tf.keras.models.load_model(weights_path, compile=False)
        model.compile(metrics=[iou_metric, top_k_metric])
        print('-------------------Start Evaluating {}-------------------'.format(weights_path))
        model.evaluate(eval_voc_ds_2012)


if __name__ == "__main__":
    eval_deeplab(os.path.join(os.getcwd(), 'deeplabv3_os8.hdf5'))
