from dataclasses import dataclass
from math import gamma
from typing import Any, Callable, Mapping, Tuple

import tensorflow as tf
import numpy as np

from ..utils import enumerate_binary_inputs
from .decoder import (
    SoftDecoderInput,
    SoftDecoderWithDelay,
    compute_bitwise_delay_llr,
    ragged_gather_nd,
    ragged_reduce_logsumexp,
)
from ..channels import NoisyChannel
from ..component import Record, RecordKey, TensorComponent, TensorRecord
from ..encoders import TrellisCode, TrellisCodeSettings


def backward_recursion_step(
    next_B: tf.Tensor,
    gamma_value_slice: tf.Tensor,
    next_states_slice: tf.Tensor,
    reducer: Callable[..., tf.Tensor],
):
    # B x |S| x 2 + B x |S| x 2 -> B x |S|
    beta = reducer(
        gamma_value_slice + tf.gather(next_B, next_states_slice, axis=1),
        2,
    )
    return beta - reducer(beta, axis=1, keepdims=True)


def backward_recursion(
    gamma_values: tf.Tensor,
    next_states: tf.Tensor,
    batch_size: int,
    K: int,
    S: int,
    reducer: Callable[..., tf.Tensor],
):
    # gamma_values = B x K x |S| x 2 : gamma_values[k, i, t] is the gamma for received k from state i to next_states[k, i, t]
    # next_states = K x |S| x 2 : next_states[k, i, t] is the next state after state i with input t at time k
    # B[k][i] = log p(Y[k+1:K-1] | s[k+1] = i)
    #         = log( Sum over t[ p(Y[k+2:K-1] | s[k+2] = next_states[k+1, i, t]) * p(Y[k+1], s[k+2] = next_states[k+1, i, t] | s[k+1] = i) ] )
    #         = logsumexp over t[ B[k+1, next_states[k+1, i, t]] + gamma_values[k+1, i, t] ]
    B = tf.TensorArray(tf.float32, size=K, clear_after_read=False)
    B = B.write(K - 1, tf.zeros((batch_size, S)))
    for k in tf.range(K - 2, -1, -1):
        # B x S
        beta = backward_recursion_step(
            B.read(k + 1), gamma_values[:, k + 1], next_states[k + 1], reducer
        )
        B = B.write(k, beta)
    return tf.transpose(B.stack(), perm=[1, 0, 2])


def forward_recursion(
    gamma_values: tf.Tensor,
    previous_states: tf.RaggedTensor,
    forward_init: tf.Tensor,
    batch_size: int,
    K: int,
    S: int,
    reducer: Callable[..., tf.Tensor],
):
    """
    gamma_values = Batch x Time x States x Inputs
    previous_states = K x |S| x |Prev| x 2 : previous_states[k, j] are the pairs of previous state that gets to j and the input to move to j at time k. |Prev| is ragged.
    forward_init = Batch x States : forward_init[m, j] is log p(s0 = j) for batch m
    A[k][j] = log p(Y[0:k-1], s[k] = j)
            = log( Sum over r[ p(Y[0:k-2], s[k-1]=previous_states[k-1, j, r, 0]) * p(Y[k-1], s[k]=j | s[k-1]=previous_states[k-1, j, r, 0]) ] )
            = logsumexp over r[ A[k-1, previous_states[k-1, j, r, 0]] + previous_gamma_values[k-1, j, r] ] ]
    logprob_values_delay = B x delay x 2 : prob_values[k, b]  = log p(u_k = b) (not u_k'). None if delay == 0
    """

    # previous_gamma_values = B x K x |S| x |Prev| : previous_gamma_values[:, k, i, t] is the gamma for received k from prev_states[i, t] to state i. |Prev| is ragged.
    # B x K x S x I ->(transpose) K x S x I x B ->(gather_nd) K x S x P (ragged) x B
    previous_gamma_values = tf.map_fn(
        ragged_gather_nd,
        elems=(tf.transpose(gamma_values, perm=[1, 2, 3, 0]), previous_states),
        fn_output_signature=tf.RaggedTensorSpec(
            shape=[S, None, batch_size], dtype=tf.float32, ragged_rank=1
        ),
    )

    A = tf.TensorArray(tf.float32, size=K, clear_after_read=False)
    # forward_init is B x S
    A = A.write(0, forward_init)
    for k in tf.range(1, K):
        # B x S
        previous_alphas: tf.Tensor = A.read(k - 1)
        # S x P (ragged) x B
        previous_gammas = previous_gamma_values[k - 1]

        alpha_not_trans = reducer(
            previous_gammas  # S x P (ragged) x B
            + tf.gather(
                tf.transpose(previous_alphas, perm=(1, 0)),  # S x B
                previous_states[k - 1, :, :, 0],  # S x P (ragged)
                axis=0,
            ),  # S x P (ragged) x B + S x P (ragged) x B
            axis=1,
        )  # S x B

        alpha_trans = tf.transpose(alpha_not_trans, perm=(1, 0))  # B x S
        A = A.write(k, alpha_trans - reducer(alpha_trans, axis=1, keepdims=True))
    return tf.transpose(A.stack(), perm=[1, 0, 2])  # B x K x |S|


def compute_delta(
    init_conditional: tf.Tensor,
    init_prior_int: tf.Tensor,
):
    """
    Parameters
    ----------
    - init_conditional = States x DelayInputs (2^delay) - This is p(s0 = s | u[0:d-1]). d := delay.
    - init_prior_int = Batch x DelayInputs (2^delay) - This is p(u[0:d-1]). d := delay.
    """
    # Softmax over DelayInputs axis
    logprob_prior_int = tf.math.log_softmax(init_prior_int, axis=1)

    # Broadcast
    # S x 2^d -> 1 x S x 2^d
    init_conditional = init_conditional[None]
    # B x 2^d -> B x 1 x 2^d
    logprob_prior_int = logprob_prior_int[:, None]

    # 1 x S x 2^d + B x 1 x 2^d -> B x S x 2^d
    return init_conditional + logprob_prior_int


# def compute_forward_prior_no_delay(batch_size: int, S: int):
#     return tf.tile(
#         tf.constant([0.0] + [-np.inf] * (S - 1))[None, :], [batch_size, 1]
#     )


def map_decode_no_delay(
    L_int: tf.Tensor,
    chi_values: tf.Tensor,
    next_states: tf.Tensor,
    previous_states: tf.RaggedTensor,
    forward_init: tf.Tensor,
    batch_size: int,
    K: int,
    S: int,
    reducer: Callable[..., tf.Tensor],
):
    L_int = tf.ensure_shape(
        L_int, [batch_size, K, 1]
    )  # For now we only support messages with one channel
    chi_values = tf.ensure_shape(chi_values, [batch_size, K, S, 2])
    next_states = tf.ensure_shape(next_states, [K, S, 2])
    forward_init = tf.ensure_shape(forward_init, [batch_size, S])

    transition_prob_values = tf.concat(
        [tf.math.log_sigmoid(-L_int), tf.math.log_sigmoid(L_int)], axis=2
    )

    # Compute ln(Gamma) values
    # gamma_values[k, i, t] = log p(Y[k], s[k+1] = next_states[k, i, t] | s[k] = i) = log p(s[k+1] = next_states[k, i, t] | s[k] = i) + chi_values[k, i, t]
    # B x K x |S| x 2
    gamma_values = chi_values + transition_prob_values[:, :, None, :]

    # Compute ln(B)
    # B x K x |S|
    B = backward_recursion(gamma_values, next_states, batch_size, K, S, reducer)

    # B x K x |S|
    A = forward_recursion(
        gamma_values, previous_states, forward_init, batch_size, K, S, reducer
    )

    # Compute L_ext
    # L = log Sum over i[ p(Y[0:K-1], s_k=i, s_k+1=next_states[k, i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], Y[k], Y[k+1:K-1], s_k=i, s_k+1=next_states[k, i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k], s_k+1=next_states[k, i, 1] | s_k=i) * P(Y[k+1:K-1] | s_k+1=next_states[k, i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k] | s_k+1=next_states[k, i, 1], s_k=i) * p(s_k+1=next_states[k, i, 1] | s_k=i) * p(Y[k+1:K-1] | s_k+1=next_states[k, i, 1]) ] / "
    # = log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k] | s_k+1=next_states[k, i, 1], s_k=i) * p(x_k=1) * p(Y[k+1:K-1] | s_k+1=next_states[k, i, 1]) ] / "
    # = log( p(x_k=1) / p(x_k=0) ) * log Sum over i[ p(Y[0:k-1], s_k=i) * p(Y[k] | s_k+1=next_states[k, i, 1], s_k=i) * p(Y[k+1:K-1] | s_k+1=next_states[k, i, 1]) ] / "
    # = L_int + logsumexp over i[ A[k, i] + chi_values[k, i, 1] + B[k, next_states[k, i, 1]] ] - logsumexp over i[ A[k, i] + chi_values[k, i, 0] + B[k, next_states[k, i, 0]] ]
    # = L_int + L_ext
    # -> L_ext = logsumexp over i[ A[k, i] + chi_values[k, i, 1] + B[k, next_states[k, i, 1]] ] - logsumexp over i[ A[k, i] + chi_values[k, i, 0] + B[k, next_states[k, i, 0]] ]

    # Should be shape B x K x |S| x 2
    # We'll need to gather for each k in [K]. Move K to first dim of B. Then use batch_dims=1
    # to specify that is the batch_dim to gather over, and use axis=2. Then we get K x B x |S| x 2.
    # Move K back to second dim.
    B_next_states = tf.transpose(
        tf.gather(tf.transpose(B, perm=[1, 0, 2]), next_states, batch_dims=1, axis=2),
        perm=[1, 0, 2, 3],
    )
    # This L_ext includes the padded bits - B x K
    L_ext = reducer(
        A + chi_values[:, :, :, 1] + B_next_states[:, :, :, 1], axis=2
    ) - reducer(A + chi_values[:, :, :, 0] + B_next_states[:, :, :, 0], axis=2)
    return tf.ensure_shape(L_ext[..., None], [batch_size, K, 1]), A, B, gamma_values


@tf.function
def map_decode(
    next_states: tf.Tensor,
    previous_states: tf.RaggedTensor,
    L_int: tf.Tensor,
    chi_values: tf.Tensor,
    init_conditional: tf.Tensor,
    L_init_int: tf.Tensor,
    delay: int,
    use_max: bool = False,
) -> tf.Tensor:
    """
    Parameters
    -----------
    - received_symbols = Batch x Time x Channels
    - next_states = Time x States x Inputs (this comes from trellis)
    - previous_states = Time x States x PrevStates x 2 (this comes from trellis)
    - L_int = Batch x Time-delay x 1 - This is the prior log likelihoods ratio that bit k+delay is 1. Trailing bits are added as 0 if there is a delay
    - chi_values = Batch x Time x States x Inputs - This comes from the channel and is the probability of
        generating received data at position k given transmitted bit at k.
    - init_conditional = States x DelayInputs (2^delay) - This is p(s0 = s | u[0:d-1]). d := delay.
    - L_init_int = Batch x DelayInputs (2^delay) - This is p(u[0:d-1]). d := delay.
    - use_max: bool - if True, then use the max approximation of logexpsum
    """
    print("Tracing map decode...")
    if use_max:
        reducer = tf.math.reduce_max
    else:
        # tf.math.reduce_logsumexp does not work with ragged tensors, use below instead
        reducer = ragged_reduce_logsumexp

    batch_size = chi_values.shape[0]
    K = chi_values.shape[1]
    S = next_states.shape[1]
    L_int = tf.ensure_shape(
        L_int, [batch_size, K - delay, 1]
    )  # For now we only support messages with one channel
    init_conditional = tf.ensure_shape(init_conditional, [S, 2**delay])
    L_init_int = tf.ensure_shape(L_init_int, [batch_size, 2**delay])

    # B x S x 2^d
    # Softmax over DelayInputs axis
    logprob_prior_int = tf.math.log_softmax(L_init_int, axis=1)
    # Broadcast
    init_conditional = init_conditional[None]  # S x 2^d -> 1 x S x 2^d
    logprob_prior_int = logprob_prior_int[:, None]  # B x 2^d -> B x 1 x 2^d
    # 1 x S x 2^d + B x 1 x 2^d -> B x S x 2^d
    forward_init = reducer(init_conditional + logprob_prior_int, axis=2)  # B x S

    # First we need to add in extra entries corresponding to 0 bits padded on because of delay
    L_int_no_delay = tf.pad(
        L_int, paddings=[[0, 0], [0, delay], [0, 0]], constant_values=-np.inf
    )

    L_ext_no_delay, A, B, gamma_values = map_decode_no_delay(
        L_int=L_int_no_delay,
        chi_values=chi_values,
        next_states=next_states,
        previous_states=previous_states,
        forward_init=forward_init,
        batch_size=batch_size,
        K=K,
        S=S,
        reducer=reducer,
    )

    # Removing the paded bits and adding channel of size 1
    if delay > 0:
        L_ext = L_ext_no_delay[:, :-delay]
    else:
        L_ext = L_ext_no_delay
    L_ext = L_ext

    L_ext = tf.ensure_shape(L_ext, [batch_size, K - delay, 1])

    # Now we compute the log posterior on the delay bits
    beta_minus1 = backward_recursion_step(
        next_B=B[:, 0],
        gamma_value_slice=gamma_values[:, 0],
        next_states_slice=next_states[0],
        reducer=reducer,
    )  # B x S
    # B x S x 1 + 1 x S x 2^d -> B x S x 2^d ->(reducer) B x 2^d
    # This is log P(Y[0:k]|u[0:d-1])
    L_init_ext = tf.math.log_softmax(
        reducer(beta_minus1[..., None] + init_conditional, axis=1), axis=1
    )  # This will just recenter things, does not affect actual formulas

    return L_ext, L_init_ext


@dataclass
class BCJRDecoderSettings:
    trellis_code: TrellisCodeSettings
    constraint: Any
    channel: Any
    use_max: bool
    name: str
    num_input_channels: int
    num_output_channels: int


class BCJRDecoder(SoftDecoderWithDelay):
    name: str = "bcjr_decoder"

    def __init__(
        self,
        encoder: TrellisCode,
        constraint: TensorComponent,
        channel: NoisyChannel,
        use_max: bool = False,
        name: str = name,
        **kwargs,
    ):
        self.name = name
        self.trellis_code = encoder
        self.constraint = constraint
        self.channel = channel
        self.use_max = use_max

        self.validate()

    def validate(self):
        assert isinstance(self.trellis_code, TrellisCode)

    @property
    def num_input_channels(self):
        return self.trellis_code.num_output_channels

    @property
    def num_output_channels(self):
        return 1

    @property
    def delay(self) -> int:
        return self.trellis_code.delay

    def call(self, data: SoftDecoderInput) -> Mapping[str, Record]:
        """
        data has 2 entries:
        "logit_prior": A 2d tensor of shape Batch x Time that holds the logit priors (log odds)
            that bit k is a 1
        "received_symbols": A 3d tensor of shape Batch x Time x Channels that holds the corrupted mesage.
        """
        received_symbols, logit_prior, logit_init_prior = (
            data["received_symbols"],
            data["logit_prior"],
            data["logit_init_prior"],
        )
        logit_bitwise_init_prior = compute_bitwise_delay_llr(
            L_init_ext=logit_init_prior,
            L_init_int=tf.zeros_like(logit_init_prior),
            delay=self.trellis_code.delay,
            use_max=self.use_max,
        )
        output_tables = self.constraint(self.trellis_code.output_tables)[
            RecordKey.OUTPUT
        ].value
        chi_values = self.channel.log_likelihood(received_symbols, output_tables)

        extrinsic_posterior, extrinsic_init = map_decode(
            next_states=self.trellis_code.trellises.state_transitions.next_states,
            previous_states=self.trellis_code.trellises.state_transitions.previous_states,
            L_int=logit_prior,
            chi_values=chi_values,
            init_conditional=self.trellis_code.init_conditional,
            L_init_int=logit_init_prior,
            delay=self.trellis_code.delay,
            use_max=self.use_max,
        )

        logit_transmitted_posterior = (
            logit_prior + extrinsic_posterior
        )  # B x K-delay x 1
        logit_bitwise_init_posterior = compute_bitwise_delay_llr(
            L_init_ext=extrinsic_init,
            L_init_int=logit_init_prior,
            delay=self.trellis_code.delay,
            use_max=self.use_max,
        )  # B x delay x 1
        # B x K x 1
        logit_posterior = tf.concat(
            [logit_bitwise_init_posterior, logit_transmitted_posterior], axis=1
        )

        return {
            RecordKey.INPUT: TensorRecord(received_symbols),
            RecordKey.LOGIT_PRIOR: TensorRecord(logit_prior),
            RecordKey.LOGIT_INIT_PRIOR: TensorRecord(logit_init_prior),
            RecordKey.LOGIT_BITWISE_INIT_PRIOR: TensorRecord(logit_bitwise_init_prior),
            RecordKey.OUTPUT: TensorRecord(logit_posterior),
            RecordKey.EXTRINSIC_POSTERIOR: TensorRecord(extrinsic_posterior),
            RecordKey.EXTRINSIC_INIT: TensorRecord(extrinsic_init),
            RecordKey.EXTRINSIC_BITWISE_INIT: TensorRecord(
                logit_bitwise_init_posterior - logit_bitwise_init_prior
            ),
        }

    def settings(self):
        return BCJRDecoderSettings(
            trellis_code=self.trellis_code.settings(),
            constraint=self.constraint.settings(),
            channel=self.channel.settings(),
            use_max=self.use_max,
            num_input_channels=self.num_input_channels,
            num_output_channels=self.num_output_channels,
            name=self.name,
        )

    def training(self):
        self.use_max = True

    def validating(self):
        self.use_max = False

    def uniform_prior_input(self, received_symbols: tf.Tensor) -> SoftDecoderInput:
        return SoftDecoderInput(
            received_symbols=received_symbols,
            logit_prior=tf.zeros(
                [
                    received_symbols.shape[0],
                    received_symbols.shape[1] - self.trellis_code.delay,
                    1,
                ]
            ),
            logit_init_prior=tf.zeros(
                [received_symbols.shape[0], 2**self.trellis_code.delay]
            ),
        )
