"""

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from nets import RetinaNet
from nets import get_backbone
from dataset import data_input_pipeline
from data_augmentation import preprocess_data
from encoding import LabelEncoder
import yaml

def train(cfg=None):
    """

    :param cfg:
    :return:
    """
    dataset_dir = cfg['DATASET_DIR']
    output_dir = cfg['OUTPUT_DIR']
    log_dir = cfg['LOGS_DIR']

    train_config = cfg['TRAIN']
    model_config = cfg['MODEL']
    solver_config = cfg['SOLVER']


    train_data_generator = data_input_pipeline(
        mode=tf.estimator.ModeKeys.TRAIN,
        dataset_dir=dataset_dir,
        preprocess_data=preprocess_data,
        batch_size=train_config['BATCH_SIZE'],
        label_encoder=LabelEncoder
    )

    input_tensor, output_tensor = get_backbone(model_config['BACKBONE'], input_shape=[640, 640, 3])

    model = RetinaNet(
        num_classes=1,
        backbone=back_bone
    )
    print(model.summary())


if __name__ == '__main__':
    config_file = './configs/retina_net_mobilenetV1.yaml'

    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    train(config)
