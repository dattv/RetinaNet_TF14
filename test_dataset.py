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
import yaml
from encoding import LabelEncoder


class test_dataset(unittest.TestCase):
    def test_coco_2007(self):
        """

        :return:
        """
        font = cv.FONT_HERSHEY_SIMPLEX
        color = (255, 0, 0)

        config_file = './configs/retina_net_mobilenetV1.yaml'

        with open(config_file) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)

        dataset_dir = cfg['DATASET_DIR']
        output_dir = cfg['OUTPUT_DIR']
        log_dir = cfg['LOGS_DIR']

        train_config = cfg['TRAIN']
        model_config = cfg['MODEL']
        solver_config = cfg['SOLVER']

        train_generator = data_input_pipeline(
            mode=tf.estimator.ModeKeys.TRAIN,
            dataset_dir=dataset_dir,
            preprocess_data=preprocess_data,
            batch_size=train_config['BATCH_SIZE'],
            label_encoder=LabelEncoder,
            config=cfg
        )

        iterator = train_generator.make_one_shot_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:
            for k in range(10):
                data = sess.run(
                    next_element
                )
                img_data = data[0]
                labels = data[1]
                for i in range(len(img_data)):
                    file_name = str(i)
                    img = cv.cvtColor(img_data[i], cv.COLOR_RGB2BGR)

                    label = labels[i]

                    bbox = label[:, :4]
                    cls = label[:, 4]

                    id = cls >= 0
                    new_cls = cls[id]
                    new_bbox = bbox[id]


                    for j in range(n_object[i]):
                        id = label[j]
                        box = bbox_data[i][j] * size_ratio[i]
                        box_int = box.astype(np.int32)
                        cv.rectangle(img, pt1=(box_int[1], box_int[0]),
                                     pt2=(box_int[3], box_int[2]), color=coco_map.COLOR_MAP[id], thickness=1)

                        text = coco_map.LABEL_MAP[id]

                        cv.putText(img, text, (box_int[1], box_int[0] + 10), font, fontScale=0.5,
                                   color=coco_map.COLOR_MAP[id], thickness=1)

                    cv.imwrite('./test_dataset_results/' + file_name, img)
            print(name)


if __name__ == '__main__':
    unittest.main()
