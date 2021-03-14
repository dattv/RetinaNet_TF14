"""

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import yaml

import tensorflow as tf
from encoding import LabelEncoder
from decoding import DecodePredictions
from dataset import data_input_pipeline
from data_augmentation import preprocess_data
from nets import get_backbone
from nets import RetinaNet
from losser import RetinaNetLoss


def train(data_dir=None,
          log_dir='./logs',
          checkpoints='./checkpoints',
          config=None):
    """

    :param data_dir:
    :param log_dir:
    :param checkpoints:
    :return:
    """

    assert data_dir != None, '{} data_dir must be a folder'.format(data_dir)
    assert config != None, '{} config must not be None'.format(config)

    train_config = config['TRAIN']
    test_config = config['TEST']
    model_config = config['MODEL']
    solver_config = config['SOLVER']
    train_config['PREPROCESS'] = preprocess_data

    if os.path.isdir(log_dir) == False:
        os.mkdir(log_dir)
    if os.path.isdir(checkpoints) == False:
        os.mkdir(checkpoints)

    model_dir = os.path.join(config['OUTPUT_DIR'], "retinanet")
    label_encoder = LabelEncoder()
    num_classes = 80

    train_dataset = data_input_pipeline(mode=tf.estimator.ModeKeys.TRAIN,
                                        dataset_dir=config['DATASET_DIR'],
                                        preprocess_data=preprocess_data,
                                        batch_size=train_config['BATCH_SIZE'],
                                        label_encoder=label_encoder)

    val_dataset = data_input_pipeline(mode=tf.estimator.ModeKeys.EVAL,
                                      dataset_dir=config['DATASET_DIR'],
                                      preprocess_data=test_config['PREPROCESS'],
                                      batch_size=test_config['BATCH_SIZE'],
                                      label_encoder=label_encoder)

    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [125, 250, 500, 240000, 360000]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    backbone = get_backbone(
        model_name=model_config['BACKBONE']
    )

    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, backbone)

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer)

    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        )
    ]

    # Uncomment the following lines, when training on full dataset
    # train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
    # val_steps_per_epoch = \
    #     dataset_info.splits["validation"].num_examples // batch_size
    #
    # train_steps = 4 * 100000
    # epochs = train_steps // train_steps_per_epoch
    #
    epochs = 1

    # Running 100 training and 50 validation steps,
    # remove `.take` when training on the full dataset

    model.fit(
        train_dataset.take(100),
        validation_data=val_dataset.take(50),
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
        initial_epoch=0
    )


def evaluation(data_dir=None,
               log_dir='./logs',
               checkpoints='./checkpoints'):
    """

    :param data_dir:
    :param log_dir:
    :param checkpoints:
    :return:
    """
    num_classes = 80
    batch_size = 8

    lastest_checkpoint = tf.train.latest_checkpoint(checkpoints)

    resnet50_backbone = get_backbone()
    model = RetinaNet(num_classes, resnet50_backbone)

    model.load_weights(lastest_checkpoint)

    # building inference model
    image = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = model(image, training=False)
    detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
    inference_model = tf.keras.Model(inputs=image, outputs=detections)

    def prepare_image(image):
        image, _, ratio = resize_and_pad_image(image, jitter=None)
        image = tf.keras.applications.resnet.preprocess_input(image)
        return tf.expand_dims(image, axis=0), ratio

    val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
    int2str = dataset_info.features["objects"]["label"].int2str


def main(data_dir=None,
         mode=tf.estimator.ModeKeys.TRAIN,
         config=None):
    """

    :param mode:
    :return:
    """
    assert data_dir != None, '{} dataset must not be None'
    assert config != None, '{} config must not be None'

    if mode == tf.estimator.ModeKeys.TRAIN:
        train(data_dir=data_dir,
              log_dir='./logs',
              checkpoints='./checkpoints',
              config=config)
    elif mode == tf.estimator.ModeKeys.EVAL:
        print('Not support yet')
    elif mode == tf.estimator.ModeKeys.PREDICT:
        print('Not support yet')


if __name__ == '__main__':

    # read config file
    f = open('./configs/retina_net_resnet50.yaml')
    config = yaml.load(f)

    main(data_dir=config['DATASET_DIR'],
         mode=tf.estimator.ModeKeys.TRAIN,
         config=config)
