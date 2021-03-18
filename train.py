"""

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os

import tensorflow as tf
import numpy as np
from nets import RetinaNet_fn
from dataset import data_input_pipeline
from data_augmentation import preprocess_data
from encoding import LabelEncoder
from losser import RetinaNetLoss
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
        label_encoder=LabelEncoder,
        config=cfg
    )

    retina_model = RetinaNet_fn(
        input_shape=[model_config['INPUT_HEIGHT'], model_config['INPUT_WIDTH'], 3],
        back_bone=model_config['BACKBONE'], num_classes=90, training=True
    )

    print(retina_model.summary())
    retina_model.save('./{}.h5'.format(retina_model.name))

    loss_fn = RetinaNetLoss(90)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    retina_model.compile(loss=loss_fn, optimizer=optimizer)

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(cfg['OUTPUT_DIR'], "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        )
    ]

    retina_model.fit(
        train_data_generator,
        epochs=10,
        callbacks=callbacks_list,
        verbose=1,
        steps_per_epoch=1000
    )


if __name__ == '__main__':
    config_file = './configs/retina_net_mobilenetV1.yaml'

    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    train(config)
