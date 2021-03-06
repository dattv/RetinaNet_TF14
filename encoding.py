"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from anchor_generator import AnchorBox
from utility import compute_iou


class LabelEncoder(object):
    """Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.

    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _match_anchor_boxes(
            self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        """Matches ground truth boxes to anchor boxes based on IOU.

        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.

        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.

        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        iou_matrix = compute_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32),
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample"""
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, sample):
        """Creates box and classification targets for a batch"""

        batch_images = sample['image']
        gt_boxes = sample["objects"]["bbox"]
        cls_ids = sample["objects"]["label"]

        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)


        batch_images = tf.keras.applications.resnet50.preprocess_input(batch_images)
        return batch_images, labels.stack()

def _compute_box_target(_box_variance, anchor_boxes, matched_gt_boxes):
    """Transforms the ground truth boxes into targets for training"""

    box_target = tf.concat(
        [
            (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
            tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
        ],
        axis=-1,
    )
    box_target = box_target / _box_variance
    return box_target


def _match_anchor_boxes(
        anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
):
    """Matches ground truth boxes to anchor boxes based on IOU.

    1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
      to get a `(M, N)` shaped matrix.
    2. The ground truth box with the maximum IOU in each row is assigned to
      the anchor box provided the IOU is greater than `match_iou`.
    3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
      box is assigned with the background class.
    4. The remaining anchor boxes that do not have any class assigned are
      ignored during training.

    Arguments:
      anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
        representing all the anchor boxes for a given input image shape,
        where each anchor box is of the format `[x, y, width, height]`.
      gt_boxes: A float tensor with shape `(num_objects, 4)` representing
        the ground truth boxes, where each box is of the format
        `[x, y, width, height]`.
      match_iou: A float value representing the minimum IOU threshold for
        determining if a ground truth box can be assigned to an anchor box.
      ignore_iou: A float value representing the IOU threshold under which
        an anchor box is assigned to the background class.

    Returns:
      matched_gt_idx: Index of the matched object
      positive_mask: A mask for anchor boxes that have been assigned ground
        truth boxes.
      ignore_mask: A mask for anchor boxes that need to by ignored during
        training
    """
    iou_matrix = compute_iou(anchor_boxes, gt_boxes)
    max_iou = tf.reduce_max(iou_matrix, axis=1)
    matched_gt_idx = tf.argmax(iou_matrix, axis=1)
    positive_mask = tf.greater_equal(max_iou, match_iou)
    negative_mask = tf.less(max_iou, ignore_iou)
    ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
    return (
        matched_gt_idx,
        tf.cast(positive_mask, dtype=tf.float32),
        tf.cast(ignore_mask, dtype=tf.float32),
    )

def encode_sample(_box_variance, anchor_boxes, gt_boxes, cls_ids):
    """
    1. convert class number to float
    2.

    :param image_shape:
    :param gt_boxes:
    :param cls_ids:
    :return:
    """

    cls_ids = tf.cast(cls_ids, dtype=tf.float32)
    matched_gt_idx, positive_mask, ignore_mask = _match_anchor_boxes(
        anchor_boxes, gt_boxes
    )
    matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
    box_target = _compute_box_target(_box_variance, anchor_boxes, matched_gt_boxes)
    matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
    cls_target = tf.where(
        tf.not_equal(positive_mask, tf.ones_like(positive_mask)), -tf.ones_like(positive_mask), matched_gt_cls_ids
    )
    cls_target = tf.where(tf.equal(ignore_mask, tf.ones_like(ignore_mask)), -2.0 * tf.ones_like(ignore_mask), cls_target)
    cls_target = tf.expand_dims(cls_target, axis=-1)
    label = tf.concat([box_target, cls_target], axis=-1)
    return label


def wraper_encode(config):
    """

    :return:
    """

    target_height = config['MODEL']['INPUT_HEIGHT']
    target_width = config['MODEL']['INPUT_WIDTH']

    anchor_generator = AnchorBox()

    anchor_boxes = anchor_generator.get_anchors(target_height, target_width)

    _box_variance = tf.convert_to_tensor(
        [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
    )

    def encode(sample):
        """

        :param sample:
        :return:
        """

        image = sample['image']
        gt_boxes = sample["objects"]["bbox"]
        cls_ids = sample["objects"]["label"]

        n_object = sample["objects"]["n_object"]

        label = encode_sample(_box_variance, anchor_boxes, gt_boxes, cls_ids)
        # batch_images = tf.keras.applications.mobilenet.preprocess_input(image)
        return image, label#, sample


    return encode

