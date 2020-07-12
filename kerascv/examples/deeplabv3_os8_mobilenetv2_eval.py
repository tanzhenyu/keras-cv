import os
import tensorflow as tf
from kerascv.data.voc_segmentation import voc_segmentation_dataset_from_directory


class MyIOUMetrics(tf.keras.metrics.Metric):
    def __init__(self, class_id, num_classes, name=None, **kwargs):
        self.num_classes = num_classes
        self.class_id = class_id
        self.acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.3)
        super(MyIOUMetrics, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(y_true, depth=self.num_classes)
        y_true = y_true[:, :, :, self.class_id]
        y_pred = y_pred[:, :, :, self.class_id]
        self.acc_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.acc_metric.result()


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
        iou_metrics = []
        for i in range(21):
            iou_metrics.append(MyIOUMetrics(class_id=i, num_classes=21, name='iou_' + str(i)))
        top_k_metric = tf.keras.metrics.SparseTopKCategoricalAccuracy()
        model = tf.keras.models.load_model(weights_path, compile=False)
        model.compile(metrics=["accuracy", top_k_metric] + iou_metrics)
        print('-------------------Start Evaluating {}-------------------'.format(weights_path))
        model.evaluate(eval_voc_ds_2012)


if __name__ == "__main__":
    eval_deeplab(os.path.join(os.getcwd(), 'deeplabv3_os8.hdf5'))
