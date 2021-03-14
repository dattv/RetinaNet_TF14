"""

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import unittest
import tensorflow as tf

from dataset import data_input_pipeline
from data_augmentation import preprocess_data
import cv2 as cv
import numpy as np


class test_dataset(unittest.TestCase):
    def test_coco_2007(self):
        """

        :return:
        """
        train_generator = data_input_pipeline(tf.estimator.ModeKeys.TRAIN,
                                              dataset_dir='/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/dataset/COCO_2',
                                              preprocess_data=None)

        iterator = train_generator.make_one_shot_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:
            img_data, bbox_data, n_object, object_type, name = sess.run([
                next_element['image'],
                next_element['objects']['bbox'],
                next_element['objects']['n_object'],
                next_element['objects']['label'],
                next_element['filename']
            ])

            for i in range(len(name)):
                file_name = str(name[i], 'utf-8')
                img = cv.cvtColor(img_data[i], cv.COLOR_RGB2BGR)

                for j in range(n_object[i]):
                    box = (bbox_data[i][j]).astype(np.uint8)
                    cv.rectangle(img, pt1=(box[1], box[0]), pt2=(box[3], box[2]), color=(0, 255, 0), thickness=2)

                cv.imwrite(file_name, img)
            print(name)


if __name__ == '__main__':
    unittest.main()
