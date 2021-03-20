"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from anchor_generator import AnchorBox
from utility import convert_to_corners_np
import numpy as np


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

class wraper_decode_np(object):
    """

    :return:
    """
    def __init__(self, config):
        target_height = config['MODEL']['INPUT_HEIGHT']
        target_width = config['MODEL']['INPUT_WIDTH']

        anchor_generator = AnchorBox(mode='numpy')

        self.anchor_boxes = anchor_generator.get_anchors(target_height, target_width)

        self._box_variance = np.asarray(
            [0.1, 0.1, 0.2, 0.2], dtype=np.float32
        )

    def _decode_box_predictions(self, index, box_predictions):

        new_anchor_boxes = self.anchor_boxes[index]
        boxes = box_predictions * self._box_variance
        boxes = np.concatenate(
            [
                boxes[..., :2] * new_anchor_boxes[..., 2:] + new_anchor_boxes[..., :2],
                np.exp(boxes[..., 2:]).astype(np.float32) * new_anchor_boxes[..., 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners_np(boxes)

        return boxes_transformed