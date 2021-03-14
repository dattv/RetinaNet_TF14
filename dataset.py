"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import zipfile
import tensorflow as tf
from data_augmentation import preprocess_data
from encoding import LabelEncoder
from utility import resize_img_keeping_ar

MAXIMUM_OBJECTS = 100
INPUT_RESOLUTION = {'height': 640,
                    'width': 640}
keras = tf.keras


def tiny_coco_2017():
    """

    :return:
    """
    url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
    filename = os.path.join(os.getcwd(), "data.zip")
    keras.utils.get_file(filename, url)

    with zipfile.ZipFile("data.zip", "r") as z_fp:
        z_fp.extractall("/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/dataset/tiny_coco_2017/")


def data_input_pipeline(mode=tf.estimator.ModeKeys.TRAIN,
                        dataset_dir='./',
                        preprocess_data=preprocess_data,
                        batch_size=8,
                        label_encoder=LabelEncoder()):
    """

    :param mode:
    :param dataset:
    :param preprocess_data:
    :param batch_size:
    :param label_encoder:
    :return:
    """

    def parse(feature):
        features = tf.io.parse_single_example(
            feature,
            features={
                'image/encoded':
                    tf.FixedLenFeature((), tf.string, default_value=''),
                'image/format':
                    tf.FixedLenFeature((), tf.string, default_value='jpeg'),
                'image/filename':
                    tf.FixedLenFeature((), tf.string, default_value=''),
                'image/key/sha256':
                    tf.FixedLenFeature((), tf.string, default_value=''),
                'image/source_id':
                    tf.FixedLenFeature((), tf.string, default_value=''),
                'image/height':
                    tf.FixedLenFeature((), tf.int64, 1),
                'image/width':
                    tf.FixedLenFeature((), tf.int64, 1),
                # Object boxes and classes.
                'image/object/bbox/xmin':
                    tf.VarLenFeature(tf.float32),
                'image/object/bbox/xmax':
                    tf.VarLenFeature(tf.float32),
                'image/object/bbox/ymin':
                    tf.VarLenFeature(tf.float32),
                'image/object/bbox/ymax':
                    tf.VarLenFeature(tf.float32),
                'image/object/class/label':
                    tf.VarLenFeature(tf.int64),
                'image/object/class/text':
                    tf.VarLenFeature(tf.string),
                'image/object/area':
                    tf.VarLenFeature(tf.float32),
                'image/object/is_crowd':
                    tf.VarLenFeature(tf.int64),
                'image/object/difficult':
                    tf.VarLenFeature(tf.int64),
                'image/object/group_of':
                    tf.VarLenFeature(tf.int64),
                'image/object/weight':
                    tf.VarLenFeature(tf.float32),

            }
        )

        # image = features['image/encoded']
        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        image = tf.reshape(image, shape=[features['image/height'], features['image/width'], 3])

        bbox = tf.concat(
            [
                tf.expand_dims(tf.sparse.to_dense(features['image/object/bbox/ymin']), axis=-1),
                tf.expand_dims(tf.sparse.to_dense(features['image/object/bbox/xmin']), axis=-1),
                tf.expand_dims(tf.sparse.to_dense(features['image/object/bbox/ymax']), axis=-1),
                tf.expand_dims(tf.sparse.to_dense(features['image/object/bbox/xmax']), axis=-1),
            ], axis=-1
        )

        image = tf.squeeze(
            tf.image.draw_bounding_boxes(
            tf.expand_dims(tf.cast(image, tf.float32), axis=0), tf.expand_dims(bbox, axis=0)
        ), axis=0)

        image, bbox = resize_img_keeping_ar(image, bbox, target_height=INPUT_RESOLUTION['height'],
                                                  target_width=INPUT_RESOLUTION['width'])


        image_id = features['image/object/class/text']
        image_name = features['image/filename']
        image_height = features['image/height']
        image_width = features['image/width']



        # org_size = [features['image/height'], features['image/width'], features['image/height'],
        #             features['image/width']]
        # bbox *= tf.cast(size_ratio, tf.float32) #/ [INPUT_RESOLUTION['height'], INPUT_RESOLUTION['width'], INPUT_RESOLUTION['height'], INPUT_RESOLUTION['width']]

        """
        Cause number of of bounding boxes in each img is variable, so we need to pad it to be a fixed size tensor
        """

        label = tf.sparse.to_dense(features['image/object/class/label'])
        object_num = tf.shape(bbox)[0]
        paddings = [[0, MAXIMUM_OBJECTS - object_num], [0, 0]]

        bbox = tf.pad(bbox, paddings)
        label = tf.pad(label, [[0, MAXIMUM_OBJECTS - object_num]])

        is_crowd = tf.sparse.to_dense(features['image/object/is_crowd'])
        is_crowd = tf.pad(is_crowd, [[0, MAXIMUM_OBJECTS - object_num]])

        source_id = features['image/source_id']
        tensor_dict = {
            'image': image,
            'height': image_height,
            'width': image_width,
            'image_id': image_id,
            'filename': image_name,
            'source_id': source_id,
            'objects': {
                'bbox': bbox,
                'n_object': object_num,
                'label': label,
                'is_crowd': is_crowd,
            },
        }

        return tensor_dict

    autotune = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(dataset_dir + '/*{}-*.tfrecord'.format(mode.lower())))
    dataset = dataset.repeat()
    dataset = dataset.map(parse)

    if preprocess_data is not None:
        dataset = dataset.map(preprocess_data, num_parallel_calls=autotune)

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(8 * batch_size)
        # dataset = dataset.map(
        #     label_encoder.encode_batch, num_parallel_calls=autotune
        # )
        # dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.prefetch(autotune)
    elif mode == tf.estimator.ModeKeys.EVAL:
        dataset = dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.prefetch(autotune)

    else:
        dataset = dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.prefetch(autotune)
    return dataset


if __name__ == '__main__':
    # tiny_coco_2017()
    data_input_pipeline(mode=tf.estimator.ModeKeys.TRAIN,
                        dataset_dir='/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/dataset/COCO_2', )
