from dataclasses import dataclass
from typing import Mapping, Tuple, TypedDict
import abc

import tensorflow as tf

from ..utils import enumerate_binary_inputs
from ..component import Component, Record, KnownChannels


# TODO: Sort out how to fill in decoder input to pass to decoders.
# The dataset has the information on what the prior should be.
# The decoder has the info on what the delay is. It can implement a method
# that takes as input the received_symbols and outputs the desired SoftDecoderInput
class SoftDecoderInput(TypedDict):
    """
    logit_prior = Batch x Time-delay - This is the prior log likelihoods ratio that bit k+delay is 1. Trailing bits are added as 0 if there is a delay
    logit_init = Batch x DelayInputs (2^delay) - This is log P(u[0:d-1]) up to additive constant. d := delay."""

    received_symbols: tf.Tensor
    logit_prior: tf.Tensor
    logit_init_prior: tf.Tensor


class SoftDecoder(Component[SoftDecoderInput], KnownChannels):
    @property
    @abc.abstractmethod
    def num_input_channels(self) -> int:
        pass

    @property
    def num_output_channels(self) -> int:
        return 1

    @abc.abstractmethod
    def call(self, data: SoftDecoderInput) -> Mapping[str, Record]:
        """
        received_symbols = Batch x Time x Channels - The 3D tensor containing the corrupted data
        logit_prior = Batch x Time - The 2D tensor containing the logit (log-odds) prior on whether
            source bit k is a 1

        Return Dict
        -----------
        `input`: `received_symbols`
        `output`: logit (log-odds) posterior whether source bit k is a 1
        """
        pass

    @abc.abstractmethod
    def uniform_prior_input(self, received_symbols: tf.Tensor) -> SoftDecoderInput:
        pass


class SoftDecoderWithDelay(SoftDecoder):
    @property
    @abc.abstractmethod
    def delay(self) -> int:
        pass


@tf.function
def ragged_gather_nd(x: Tuple[tf.Tensor, tf.Tensor]):
    print("Tracing ragged_gather_nd...")
    return tf.gather_nd(x[0], x[1])


@tf.function
def ragged_reduce_logsumexp(arr, axis, keepdims=False):
    print("Tracing ragged_reduce_logsumexp")
    raw_max = tf.reduce_max(arr, axis=axis, keepdims=True)
    my_max = tf.stop_gradient(
        tf.where(tf.math.is_finite(raw_max), raw_max, tf.zeros_like(raw_max))
    )
    result = (
        tf.math.log(
            tf.math.reduce_sum(tf.math.exp(arr - my_max), axis=axis, keepdims=True)
        )
        + my_max
    )
    # Reduce sum over a single item axis to apply keepdims conditionally without losing shape inference
    return tf.math.reduce_sum(result, axis=axis, keepdims=keepdims)


def compute_bitwise_delay_llr(
    L_init_ext: tf.Tensor, L_init_int: tf.Tensor, delay: int, use_max: bool
):
    if use_max:
        reducer = tf.math.reduce_max
    else:
        reducer = ragged_reduce_logsumexp

    batch_size = L_init_ext.shape[0]
    L_init_ext = tf.ensure_shape(L_init_ext, [batch_size, 2**delay])
    L_init_int = tf.ensure_shape(L_init_int, [batch_size, 2**delay])
    L_init = L_init_int + L_init_ext  # B x 2^d

    # 2^d x d
    delay_inputs = tf.cast(enumerate_binary_inputs(delay), dtype=tf.float32)

    # B x 2^d x 1 + 1 x 2^d x d -> B x 2^d x d ->(reducer) B x d
    logit_posterior = reducer(
        tf.math.log_softmax(L_init[..., None]) + tf.math.log(delay_inputs)[None], axis=1
    )

    return tf.ensure_shape(logit_posterior[..., None], [batch_size, delay, 1])


def compute_delay_logit_from_bitwise(L_init_bitwise: tf.Tensor):
    """
    - L_init_bitwise = Batch x delay x 1
    """
    # Now Batch x delay
    L_init_bitwise = tf.ensure_shape(L_init_bitwise, [None, None, 1])[..., 0]
    delay = L_init_bitwise.shape[1]

    # 2^d x d
    delay_inputs = tf.cast(enumerate_binary_inputs(delay), dtype=tf.float32)
    multiplier = (
        2.0 * delay_inputs - 1.0
    )  # 1 if bit is 1, otherwise flip bitwise logit by -1

    # B x d @ d x 2^d -> B x 2^d
    L_init = tf.matmul(L_init_bitwise, tf.transpose(multiplier, perm=[1, 0]))

    return L_init
