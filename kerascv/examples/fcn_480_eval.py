import os
import tensorflow as tf
from kerascv.data.voc_segmentation import voc_segmentation_dataset_from_directory


def eval_fcn(weights_path):
    batch_size = 20
    if os.path.exists(weights_path):
        eval_voc_ds_2012 = voc_segmentation_dataset_from_directory(
            split="val", batch_size=batch_size)
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            iou_metric = tf.keras.metrics.MeanIoU(num_classes=21)
            model = tf.keras.models.load_model(weights_path, compile=False)
            y_pred = model.outputs[0]
            y_pred = tf.math.argmax(y_pred, axis=-1)
            inputs = model.inputs
            eval_model = tf.keras.Model(inputs, y_pred)
            eval_model.compile(metrics=[iou_metric])
        print('-------------------Start Evaluating {}-------------------'.format(weights_path))
        eval_model.evaluate(eval_voc_ds_2012)


if __name__ == "__main__":
    eval_fcn(os.path.join(os.getcwd(), 'fcn_32.hdf5'))
    eval_fcn(os.path.join(os.getcwd(), 'fcn_16.hdf5'))
    eval_fcn(os.path.join(os.getcwd(), 'fcn_8.hdf5'))
