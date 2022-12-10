from dataclasses import dataclass
import logging
from typing import Mapping, Sequence, Tuple
import math

import tensorflow as tf
import numpy as np

from ..constants import EPSILON
from ..utils import bitarray2dec, enumerate_binary_inputs
from ..component import (
    TensorComponent,
    TensorRecord,
    Record,
    RecordKey,
)
from ..trellis import (
    TrainableUnrolledTrellis,
    UnrolledStateTransitionGraph,
    UnrolledTrellis,
)
from .encoder import Encoder

"""
Binary numbers are represented as big endian unless otherwise stated
Convolutional code state transition drops leftmost bit (most significant if big endian)
"""


@dataclass
class TrellisCodeSettings:
    trellises: UnrolledTrellis
    name: str
    normalize_output_table: bool
    delay_state_transitions: UnrolledStateTransitionGraph
    num_input_channels: int
    num_output_channels: int
    num_states: int
    delay: int


class TrellisCode(Encoder):
    name: str = "trellis_code"

    def __init__(
        self,
        trellises: UnrolledTrellis,
        name: str = name,
        normalize_output_table=False,
        delay_state_transitions: UnrolledStateTransitionGraph = None,
    ):
        self.name = name
        self.trellises = trellises
        self.normalize_output_table = normalize_output_table
        self.delay_state_transitions = delay_state_transitions

        self.validate()

        (
            self.init_conditional,
            self.delayed_init_state,
        ) = self._construct_init_conditional()

    def validate(self):
        if self.delay_state_transitions is not None:
            assert self.delay_state_transitions.num_states == self.trellises.num_states

    @property
    def num_states(self) -> int:
        return self.trellises.num_states

    @property
    def output_tables(self):
        output_tables = self.trellises.output_tables
        if self.normalize_output_table:
            output_tables = (
                output_tables
                - tf.reduce_mean(output_tables, axis=[1, 2], keepdims=True)
            ) / (
                EPSILON + tf.math.reduce_std(output_tables, axis=[1, 2], keepdims=True)
            )
            return output_tables
        else:
            return output_tables

    @property
    def delay(self) -> int:
        return (
            0
            if self.delay_state_transitions is None
            else self.delay_state_transitions.num_steps
        )

    def _construct_init_conditional(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Returns
        -------
        - init_conditional = States x DelayInputs (2^delay) - This is log p(s0 = s | u[0:d-1]). d := delay.
            if delay == 0: This is just log p(s0 = s) = log [1 if s0 = 0; 0 otherwise]
                                                      = 0 if s0 = 0; -inf otherwise
        - delayed_init_state = DelayInputs (2^delay) - This maps each delay input to a starting state.

        """
        if self.delay_state_transitions is None:
            # delay = 0, so shape is States x 1
            init_conditional = tf.constant([0.0] + [-np.inf] * (self.num_states - 1))[
                :, None
            ]
            # delay = 0 so shape is 2 ^ 0 = 1
            delayed_init_state = tf.constant([0])
        else:
            # 2^delay x delay
            binary_inputs = enumerate_binary_inputs(self.delay)
            num_delay_inputs = binary_inputs.shape[0]
            # TensorArray will be 1 x 2^delay - tensors are not stacked, discarded after reading.
            # Just for accumulating state
            states: tf.TensorArray = tf.TensorArray(size=1, dtype=tf.int32)
            states = states.write(0, tf.zeros(num_delay_inputs, dtype=tf.int32))
            for t in tf.range(self.delay):
                # 2^delay x 2
                ind_tensor = tf.stack([states.read(0), binary_inputs[:, t]], axis=1)
                states = states.write(
                    0,
                    tf.gather_nd(
                        self.delay_state_transitions.next_states[t], ind_tensor
                    ),
                )
            # 2^delay
            delayed_init_state = states.read(0)

            # Create scatter indices -> 2^delay x 2
            scatter_indicies = tf.stack(
                [delayed_init_state, tf.range(num_delay_inputs)], axis=1
            )
            # Scatters into States x 2^delay
            init_conditional = tf.math.log(
                tf.scatter_nd(
                    scatter_indicies,
                    tf.ones((num_delay_inputs,)),
                    shape=(self.num_states, num_delay_inputs),
                )
            )

        assert init_conditional.shape[0] == self.num_states
        assert init_conditional.shape[1] == 2**self.delay
        assert delayed_init_state.shape[0] == 2**self.delay
        return init_conditional, delayed_init_state

    def call(self, data: tf.Tensor) -> Mapping[str, Record]:

        """
        Assumes message is set of binary streams. Each channel is an individual stream
        `data` comes in as shape Batch x Time x Channels.
        """
        # Turned into Batch x Time. `reduced` means the binary channels have been collapsed into decimal number
        msg_reduced_original = bitarray2dec(tf.cast(data, tf.int32), axis=2)
        batch_size = msg_reduced_original.shape[0]
        msg_len = msg_reduced_original.shape[1]
        # TensorArray will be Time x Batch x Channels
        output: tf.TensorArray = tf.TensorArray(size=msg_len, dtype=tf.float32)
        # TensorArray will be Time x Batch - tensors are not stacked, discarded after reading
        states: tf.TensorArray = tf.TensorArray(size=msg_len + 1, dtype=tf.int32)
        states = states.write(0, tf.zeros(batch_size, dtype=tf.int32))
        # Run the delay out - TODO: I can now use the delay_init_states instead of running the delay out.
        if self.delay_state_transitions is not None:
            for t in tf.range(self.delay):
                ind_tensor = tf.stack(
                    [states.read(0), msg_reduced_original[:, t]], axis=1
                )
                states = states.write(
                    0,
                    tf.gather_nd(
                        self.delay_state_transitions.next_states[t], ind_tensor
                    ),
                )
        # Pad the end with 0's to make up for the delay - msg_reduced.shape[0] == msg_len
        msg_reduced = tf.pad(
            msg_reduced_original[:, self.delay :], [[0, 0], [0, self.delay]], "CONSTANT"
        )

        output_tables, next_states = self.trellises.ensure_length(msg_reduced.shape[1])
        for t in tf.range(msg_len):
            ind_tensor = tf.stack([states.read(t), msg_reduced[:, t]], axis=1)
            output = output.write(t, tf.gather_nd(output_tables[t], ind_tensor))
            states = states.write(t + 1, tf.gather_nd(next_states[t], ind_tensor))
        states = states.close()
        # Turn the outputs from Time x Batch x Channels to Batch x Time x Channels
        output_result = tf.transpose(output.stack(), perm=[1, 0, 2])
        return {
            RecordKey.INPUT: TensorRecord(data),
            RecordKey.OUTPUT: TensorRecord(output_result),
        }

    def check_trellis_code_compatibility(self, other: "TrellisCode") -> bool:
        return (
            self.delay_state_transitions == other.delay_state_transitions
            and self.trellises.check_state_table_compatibility(other.trellises)
            and self.normalize_output_table == other.normalize_output_table
        )

    def concat(self, code2: TensorComponent) -> TensorComponent:
        if isinstance(code2, TrellisCode):
            if self.check_trellis_code_compatibility(code2):
                concat_trellis = self.trellises.concat_outputs(
                    code2.trellises, check_compatibility=False
                )  # We already checked
                return TrellisCode(
                    concat_trellis,
                    name="_".join([self.name, code2.name]),
                    normalize_output_table=(self.normalize_output_table),
                    delay_state_transitions=self.delay_state_transitions,
                )
            else:
                logging.warning(
                    f"Trellis codes {self.name} and {code2.name} were not compatible. Falling back to generic code concatenation."
                )
                return super().concat(code2)
        else:
            return super().concat(code2)

    def with_systematic(self) -> "TrellisCode":
        return TrellisCode(
            self.trellises.with_systematic(),
            name="_".join([self.name, "systematic"]),
            normalize_output_table=self.normalize_output_table,
            delay_state_transitions=self.delay_state_transitions,
        )

    def __mul__(self, other: object):
        return TrellisCode(
            self.trellises * other,
            name=self.name,
            normalize_output_table=self.normalize_output_table,
            delay_state_transitions=self.delay_state_transitions,
        )

    def __add__(self, other: object):
        return TrellisCode(
            self.trellises + other,
            name=self.name,
            normalize_output_table=self.normalize_output_table,
            delay_state_transitions=self.delay_state_transitions,
        )

    def __sub__(self, other: object):
        return TrellisCode(
            self.trellises - other,
            name=self.name,
            normalize_output_table=self.normalize_output_table,
            delay_state_transitions=self.delay_state_transitions,
        )

    def __truediv__(self, other: object):
        return TrellisCode(
            self.trellises / other,
            name=self.name,
            normalize_output_table=self.normalize_output_table,
            delay_state_transitions=self.delay_state_transitions,
        )

    def settings(self) -> TrellisCodeSettings:
        return TrellisCodeSettings(
            trellises=self.trellises,
            name=self.name,
            normalize_output_table=self.normalize_output_table,
            delay_state_transitions=self.delay_state_transitions,
            num_input_channels=self.num_input_channels,
            num_output_channels=self.num_output_channels,
            num_states=self.trellises.num_states,
            delay=self.delay,
        )

    def training(self):
        raise NotImplementedError()
        # self.trellises = self.trellises.training()

    def validating(self):
        self.trellises = self.trellises.validating()

    @property
    def num_input_channels(self):
        return math.ceil(math.log2(self.trellises.num_inputs))

    @property
    def num_output_channels(self):
        return self.trellises.num_outputs

    @property
    def num_steps(self):
        return self.trellises.num_steps

    def parameters(self) -> Sequence[tf.Variable]:
        if isinstance(self.trellises, TrainableUnrolledTrellis):
            return [self.trellises.output_tables]
        else:
            return []
