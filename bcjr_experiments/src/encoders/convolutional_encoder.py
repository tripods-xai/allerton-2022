import logging
import math
from typing import Optional

import tensorflow as tf

from ..trellis import Trellis, StateTransitionGraph
from .trellis_encoder import TrellisCode
from ..utils import (
    check_int,
    base_2_accumulator,
    dec2bitarray,
    bitarray2dec,
    enumerate_binary_inputs,
)


class GeneralizedConvolutionalCode(TrellisCode):
    """
    Convolutional code represented by a table
    feedback table is required to be a binary table
    """

    name: str = "generalized_convolutional_code"

    def __init__(
        self,
        table: tf.Tensor,
        num_steps: int,
        feedback: tf.Tensor = None,
        name: str = name,
        delay=0,
    ):
        """
        Parameters
        ---------
        table = Inputs x Channels - Maps possible (binary) inputs to the output message
        feedback = Inputs - feedback for each input
        """
        self.table = table
        self.feedback = feedback

        self.window = check_int(math.log2(self.num_possible_windows))
        self._base_2 = base_2_accumulator(self.window)

        self.trellis = self._construct_trellis(
            table=self.table,
            window=self.window,
            num_possible_windows=self.num_possible_windows,
            feedback=self.feedback,
        )
        if delay > 0:
            delay_state_transitions = self.trellis.state_transitions.unroll(delay)
        else:
            delay_state_transitions = None
        super().__init__(
            trellises=self.trellis.unroll(num_steps),
            name=name,
            normalize_output_table=False,
            delay_state_transitions=delay_state_transitions,
        )

        self.validate()

    @property
    def num_possible_windows(self) -> int:
        return self.table.shape[0]

    def validate(self):
        # These convolutional codes are fixed to only take in 1 input bit
        assert self.num_input_channels == 1
        if self.feedback is not None:
            assert self.num_possible_windows == self.feedback.shape[0]
            tf.debugging.assert_type(self.feedback, tf_type=tf.int32)
        # Debugging type checks
        tf.debugging.assert_type(self.table, tf_type=tf.float32)

    @staticmethod
    def _construct_trellis(
        table: tf.Tensor,
        window: int,
        num_possible_windows: int,
        feedback: Optional[tf.Tensor],
    ) -> Trellis:
        binary_states = dec2bitarray(tf.range(num_possible_windows), num_bits=window)
        if feedback is not None:
            binary_states = tf.concat(
                [binary_states[:, :-1], feedback[:, None]], axis=1
            )

        # Even indicies correspond to the last bit not included in state.
        # (i.e. input is coming from the right when reading data left to right)
        reordered_table = tf.gather(table, bitarray2dec(binary_states))
        output_table = tf.stack([reordered_table[::2], reordered_table[1::2]], axis=1)

        next_states = bitarray2dec(binary_states[:, 1:], axis=-1)
        next_states_table = tf.stack([next_states[::2], next_states[1::2]], axis=1)

        return Trellis(
            state_transitions=StateTransitionGraph.from_next_states(next_states_table),
            output_table=output_table,
        )

    def _check_recursive_condition(self):
        # TODO Figure out how to do rsc for composite trellis codes
        if self.feedback is not None:
            raise ValueError(f"Cannot create recursive code, code already has feedback")
        if not tf.reduce_all(
            tf.logical_or(
                self.trellis.output_table[:, :, 0] == 0.0,
                self.trellis.output_table[:, :, 0] == 1.0,
            )
        ):
            raise ValueError(f"First channel is not binary out.")
        if not tf.reduce_all(
            self.trellis.output_table[:, 0, 0] != self.trellis.output_table[:, 1, 0]
        ):
            raise ValueError(
                f"Cannot invert code, some output does not change when input changes: 0 -> {self.trellis.output_table[:, 0, 0]} 1 -> {self.trellis.output_table[:, 1, 0]}"
            )

    def to_rc(self):
        self._check_recursive_condition()
        feedback = tf.cast(self.table[:, 0], dtype=tf.int32)
        return GeneralizedConvolutionalCode(
            table=self.table[:, 1:], num_steps=self.num_steps, feedback=feedback
        )

    def to_rsc(self):
        return self.to_rc().with_systematic()


class AffineConvolutionalCode(GeneralizedConvolutionalCode):
    """Convolutional code represented by a single boolean affine function"""

    name: str = "affine_convolutional_code"

    def __init__(
        self,
        generator: tf.Tensor,
        bias: tf.Tensor,
        num_steps: int,
        name: str = name,
        delay=0,
    ):
        """
        generator = Channels x Width - Channels is output channels, and Width corresponds to window
        bias = Channels - Channels is output channels. Binary bias on code.
        """
        self.validate_inputs(generator, bias)

        self.generator = generator
        self.bias = bias

        window = self.generator.shape[1]

        self._generator_filter = tf.cast(
            tf.transpose(self.generator, perm=[1, 0])[:, None, :], dtype=tf.float32
        )  # Width x InChannels=1 x Channels

        # Create the table
        self.code_inputs = enumerate_binary_inputs(window)  # Batch=Inputs x Time
        table = self._encode(self.code_inputs)[:, 0, :]  # Inputs x Channels

        super().__init__(table, num_steps=num_steps, name=name, delay=delay)

    def validate_inputs(self, generator, bias):
        tf.debugging.assert_type(generator, tf_type=tf.int32)
        tf.debugging.assert_type(bias, tf_type=tf.int32)
        assert generator.shape[0] == bias.shape[0]

    def _encode(self, msg: tf.Tensor) -> tf.Tensor:
        """
        msg = Batch x Time - binary message to encode

        Returns
        -------
        Batch x Time x Channels - encoded binary message

        Actual logic for encoding, is not responsible for padding with initial state.
        Assumes that the message only has 1 channel (and thus channel dimension is reduced out).
        """
        # Convolve generator over message to efficiently get outputs
        conv_msg_input = tf.cast(
            msg[:, :, None], dtype=tf.float32
        )  # Batch x Time x InChannels
        conv_output = tf.cast(
            tf.nn.conv1d(
                conv_msg_input, self._generator_filter, stride=1, padding="VALID"
            ),
            dtype=tf.int32,
        )  # Batch x Time x Channels
        return tf.cast((conv_output + self.bias[None, None, :]) % 2, dtype=tf.float32)
