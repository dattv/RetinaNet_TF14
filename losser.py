"""

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class RetinaNetBoxLoss(tf.keras.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, delta):
        super(RetinaNetBoxLoss, self).__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(tf.keras.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma):
        super(RetinaNetClassificationLoss, self).__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)

        alpha = tf.where(
            tf.equal(y_true, tf.ones_like(y_true)),
            self._alpha * tf.ones_like(y_true),
            (1.0 - self._alpha)*tf.ones_like(y_true)
        )

        pt = tf.where(
            tf.equal(y_true, tf.ones_like(y_true)),
            probs * tf.ones_like(y_true),
            (1.0 - probs) * tf.ones_like(y_true)
        )
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
        # return tf.reduce_sum(cross_entropy, axis=-1)


class RetinaNetLoss(tf.keras.losses.Loss):
    """Wrapper to combine both the losses"""

    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0, delta=1.0):
        super(RetinaNetLoss, self).__init__(reduction="auto", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]

        # cls_labels = y_true[:, :, 4]
        # cls_labels = tf.Print(cls_labels, [cls_labels], message='cls_labels')
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)

        # sum_possitive_mask = tf.reduce_sum(positive_mask)
        # sum_possitive_mask = tf.Print(sum_possitive_mask, [sum_possitive_mask], message='sum_possitive_mask')

        # sum_ignore_mask = tf.reduce_sum(ignore_mask)
        # sum_ignore_mask = tf.Print(sum_ignore_mask, [sum_ignore_mask], message='sum_ignore_mask')

        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)

        clf_loss = tf.where(tf.equal(ignore_mask, tf.ones_like(ignore_mask)), tf.zeros_like(clf_loss), clf_loss)

        box_loss = tf.where(tf.equal(positive_mask, tf.ones_like(positive_mask)), box_loss, tf.zeros_like(positive_mask))

        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        # normalizer = tf.Print(normalizer, [normalizer], message='normalizer')
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        clf_loss = tf.Print(clf_loss, [clf_loss], message='clf_loss')
        box_loss = tf.Print(box_loss, [box_loss], message='box_loss')
        loss = clf_loss + box_loss
        return loss
