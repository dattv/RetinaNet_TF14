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
from decoding import wraper_decode_np


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

        decode_layer_np = wraper_decode_np(config=cfg)

        with tf.Session() as sess:
            for k in range(100):
                data = sess.run(
                    next_element
                )
                img_data = data[0]
                labels = data[1]
                # sample = data[2]
                for i in range(len(img_data)):
                    file_name = str(k) + "_" + str(i)
                    img = cv.cvtColor(img_data[i], cv.COLOR_RGB2BGR)

                    label = labels[i]

                    bbox = label[:, :4]

                    cls = label[:, 4]

                    id = cls >= 0
                    new_cls = cls[id]
                    new_bbox = bbox[id]

                    new_bbox = decode_layer_np._decode_box_predictions(id, new_bbox)
                    new_anchor = decode_layer_np.anchor_boxes[id]
                    for j in range(len(new_bbox)):
                        id = new_cls[j]
                        box = new_bbox[j]
                        box_int = box.astype(np.int32)

                        anchor = new_anchor[j]
                        # cv.rectangle(img, pt1=(box_int[0], box_int[1]),
                        #              pt2=(box_int[2], box_int[3]), color=coco_map.COLOR_MAP[id], thickness=1)

                        cv.rectangle(img,
                                     pt1=(int(anchor[0] - anchor[2] / 2),
                                               int(anchor[1] - anchor[3] / 2)),
                                     pt2=(int(anchor[0] + anchor[2] / 2),
                                          int(anchor[1] + anchor[3] / 2)),
                                     color=coco_map.COLOR_MAP[id], thickness=1)

                        text = coco_map.LABEL_MAP[new_cls[j]]

                        # cv.putText(img, text, (box_int[0], box_int[1] + 10), font, fontScale=0.5,
                        #            color=coco_map.COLOR_MAP[id], thickness=1)

                    # n_object = sample['objects']['n_object'][i]
                    # label = sample['objects']['label'][i]
                    # bbox = sample['objects']['bbox'][i]
                    #
                    # for j in range(n_object):
                    #     temp_labels = label[j]
                    #     text = coco_map.LABEL_MAP[temp_labels]
                    #     temp_box = bbox[j].astype(np.int32)
                    #
                    #     cv.rectangle(img, pt1=(temp_box[0] - temp_box[2] // 2, temp_box[1] - temp_box[3] // 2),
                    #                  pt2=(temp_box[0] + temp_box[2] // 2, temp_box[1] + temp_box[3] // 2), color=coco_map.COLOR_MAP[temp_labels], thickness=1)
                    #
                    #     cv.putText(img, text, (temp_box[0] - temp_box[2] // 2, temp_box[1] - temp_box[3] // 2 + 10), font, fontScale=0.5,
                    #                color=coco_map.COLOR_MAP[temp_labels], thickness=2)

                    name = './test_dataset_results/' + file_name + '.jpg'
                    cv.imwrite(name, img)
                    print(name)


if __name__ == '__main__':
    unittest.main()
