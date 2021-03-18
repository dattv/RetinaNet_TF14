"""

"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from encoding import AnchorBox
from utility import convert_to_corners

keras = tf.keras

def RetinaNet_fn(input_shape=[224, 224, 3], back_bone='resnet50', num_classes=90, training=True):
    """

    :param input_shape:
    :param back_bone:
    :param training:
    :return:
    """
    # Backbone
    if back_bone.lower() == 'resnet50':
        """Builds ResNet50 with pre-trained imagenet weights"""
        backbone = keras.applications.ResNet50(
            include_top=False, input_shape=input_shape
        )

        layer_names = ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]

    else:
        """Builds mobilenetV1 with pre-trained imagenet weights"""
        backbone = keras.applications.MobileNet(
            include_top=False, input_shape=input_shape
        )
        layer_names = ["conv_pw_5_relu", "conv_pw_11_relu", "conv_pw_13_relu"]

    for layer in backbone.layers:
        layer.trainable = training

    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in layer_names
    ]

    input = backbone.input

    # FeaturePyramid
    p3_output = tf.keras.layers.Conv2D(256, 1, 1, "same")(c3_output)
    p4_output = tf.keras.layers.Conv2D(256, 1, 1, "same")(c4_output)
    p5_output = tf.keras.layers.Conv2D(256, 1, 1, "same")(c5_output)

    p4_output = tf.keras.layers.add(
        [p4_output, tf.keras.layers.UpSampling2D(2)(p5_output)]
    )
    p3_output = tf.keras.layers.add(
        [p3_output, tf.keras.layers.UpSampling2D(2)(p4_output)]
    )
    p3_output = tf.keras.layers.Conv2D(256, 3, 1, "same")(p3_output)
    p4_output = tf.keras.layers.Conv2D(256, 3, 1, "same")(p4_output)
    p5_output = tf.keras.layers.Conv2D(256, 3, 1, "same")(p5_output)

    p6_output = tf.keras.layers.Conv2D(256, 3, 2, "same")(c5_output)
    p7_output = tf.keras.layers.Conv2D(256, 3, 2, "same")(
        tf.keras.layers.Activation('relu')(p6_output)
    )

    FPN = [p3_output, p4_output, p5_output, p6_output, p7_output]

    # Head
    prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))


    cls_outputs = []
    box_outputs = []
    for feature in FPN:

        box_outputs.append(tf.keras.layers.Reshape(target_shape=[-1, 4])(
            build_head_fn(feature, 9 * num_classes, prior_probability)))
        cls_outputs.append(
            tf.keras.layers.Reshape(target_shape=[-1, num_classes])(build_head_fn(feature, 9 * 4, "zeros"))
        )

    cls_outputs = tf.keras.layers.Concatenate(axis=1)(cls_outputs)
    box_outputs = tf.keras.layers.Concatenate(axis=1)(box_outputs)

    output = tf.keras.layers.Concatenate(axis=-1)([cls_outputs, box_outputs])
    return tf.keras.Model(inputs=input, outputs=output, name='RetinaNet')

def build_head_fn(input, output_filters, bias_init):
    """

    :param input:
    :param output_filters:
    :param bias_init:
    :return:
    """
    output = input
    kernel_init = tf.initializers.random_normal(0.0, 0.01)
    for _ in range(4):
        output = tf.keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)(output)
        output = tf.keras.layers.Activation('relu')(output)

    output = tf.keras.layers.Conv2D(
        output_filters, 3, 1, padding="same", kernel_initializer=kernel_init, bias_initializer=bias_init
    )(output)

    return output


class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
            self,
            num_classes=80,
            confidence_threshold=0.05,
            nms_iou_threshold=0.5,
            max_detections_per_class=100,
            max_detections=100,
            box_variance=[0.1, 0.1, 0.2, 0.2],
            **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )
