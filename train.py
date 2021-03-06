"""

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
from datetime import datetime
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
    test_config = cfg['TEST']
    model_config = cfg['MODEL']
    solver_config = cfg['SOLVER']

    n_classes = model_config['NUM_CLASSES']

    train_data_generator = data_input_pipeline(
        mode=tf.estimator.ModeKeys.TRAIN,
        dataset_dir=dataset_dir,
        preprocess_data=preprocess_data,
        batch_size=train_config['BATCH_SIZE'],
        label_encoder=LabelEncoder,
        config=cfg
    )

    test_data_generator = data_input_pipeline(
        mode=tf.estimator.ModeKeys.EVAL,
        dataset_dir=dataset_dir,
        batch_size=test_config['BATCH_SIZE'],
        label_encoder=LabelEncoder,
        config=config
    )

    retina_model = RetinaNet_fn(
        input_shape=[model_config['INPUT_HEIGHT'], model_config['INPUT_WIDTH'], 3],
        back_bone=model_config['BACKBONE'], num_classes=n_classes, training=True
    )

    print(retina_model.summary())
    tf.keras.utils.plot_model(retina_model, to_file='retina.png', show_shapes=True)

    retina_model.save('./{}.h5'.format(retina_model.name))

    loss_fn = RetinaNetLoss(num_classes=n_classes)

    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [125., 250., 500., 240000., 360000.]
    learning_rate_fn = tf.compat.v2.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    retina_model.compile(loss=loss_fn, optimizer=optimizer)

    logdir = os.path.join(config['LOGS_DIR'], datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(cfg['OUTPUT_DIR'], "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=logdir, write_images=True, profile_batch=10
        ),
    ]

    retina_model.fit(
        train_data_generator,
        epochs=train_config['N_EPOCH'],
        callbacks=callbacks_list,
        verbose=1,
        steps_per_epoch=118200 // train_config['BATCH_SIZE'],
        validation_data=test_data_generator.take(50)
    )


if __name__ == '__main__':
    config_file = './configs/retina_net_mobilenetV1.yaml'

    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    train(config)
