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
import coco_map


class test_dataset(unittest.TestCase):
    def test_coco_2007(self):
        """

        :return:
        """
        font = cv.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 0)

        train_generator = data_input_pipeline(tf.estimator.ModeKeys.TRAIN,
                                              dataset_dir='/media/dat/68fa98f8-9d03-4c1e-9bdb-c71ea72ab6fa/dat/dataset/COCO_2',
                                              preprocess_data=None,
                                              batch_size=1)

        iterator = train_generator.make_one_shot_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:
            for k in range(10):
                img_data, bbox_data, n_object, object_type, name, height, width, labels, size_ratio = sess.run([
                    next_element['image'],
                    next_element['objects']['bbox'],
                    next_element['objects']['n_object'],
                    next_element['objects']['label'],
                    next_element['filename'],
                    next_element['height'],
                    next_element['width'],
                    next_element['objects']['label'],
                    next_element['size_ratio']
                ])

                for i in range(len(name)):
                    file_name = str(name[i], 'utf-8')
                    img = cv.cvtColor(img_data[i], cv.COLOR_RGB2BGR)

                    print('height: ', height[i])
                    print('width: ', width[i])
                    label = labels[i]
                    for j in range(n_object[i]):
                        id = label[j]
                        box = bbox_data[i][j] * size_ratio[i]
                        box_int = box.astype(np.int32)
                        cv.rectangle(img, pt1=(box_int[1], box_int[0]),
                                     pt2=(box_int[3], box_int[2]), color=coco_map.COLOR_MAP[id], thickness=1)


                        text = coco_map.LABEL_MAP[id]

                        cv.putText(img, text, (box_int[1], box_int[0] + 10), font, fontScale=0.5, color=coco_map.COLOR_MAP[id], thickness=1)

                    cv.imwrite('./test_dataset_results/' + file_name, img)
            print(name)


if __name__ == '__main__':
    unittest.main()
