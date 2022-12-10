from typing import TypedDict

import tensorflow as tf


BitErrorMetrics = TypedDict("BitErrorMetrics", {"ber": tf.Tensor, "bler": tf.Tensor})


def bit_error_metrics(
    original_msg: tf.Tensor, msg_confidence: tf.Tensor
) -> BitErrorMetrics:
    """
    original_msg = Batch x Time - original transmitted binary message
    msg_confidence = Batch x Time - logit (log-odd) posterior guess for message bits.
    """
    ber_for_block = tf.reduce_mean(
        tf.cast(
            tf.not_equal(original_msg, tf.cast(msg_confidence > 0, tf.float32)),
            tf.float32,
        ),
        axis=[1, 2],
    )
    block_error_for_block = tf.cast(ber_for_block > 0, tf.float32)
    return {"ber": ber_for_block, "bler": block_error_for_block}


def cross_entropy_with_logits(
    original_msg: tf.Tensor,
    msg_confidence: tf.Tensor,
) -> tf.Tensor:
    """
    original_msg = Batch x Time - original transmitted binary message
    msg_confidence = Batch x Time - logit (log-odd) posterior guess for message bits.
    """
    return tf.nn.sigmoid_cross_entropy_with_logits(
        labels=original_msg, logits=msg_confidence
    )


def kl_divergence(logit_categorical_a, logit_categorical_b):
    """
    - logit_cateogrical_a = Batch x 2^d - up to additive constant
    - logit_cateogrical_b = Batch x 2^d - up to additive constant
    """
    log_prob_a = tf.math.log_softmax(logit_categorical_a, axis=1)
    prob_a = tf.math.exp(log_prob_a)
    log_prob_b = tf.math.log_softmax(logit_categorical_b, axis=1)

    # B x 2^d * B x 2^d ->(sum) B
    return tf.reduce_sum(prob_a * (log_prob_a - log_prob_b), axis=1)
