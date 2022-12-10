from dataclasses import dataclass
from typing import List, Sequence, Tuple

import tensorflow as tf
import numpy as np


@dataclass
class StateTransitionGraph:
    next_states: tf.Tensor
    # RaggedTensor of |States| x |PrevStates| x 2. Last dimension is pair previous state and transition input
    previous_states: tf.RaggedTensor

    def __post_init__(self):
        tf.debugging.assert_type(self.next_states, tf_type=tf.int32)
        tf.debugging.assert_rank(self.next_states, 2)
        assert self.previous_states.dtype == tf.int32
        assert self.previous_states.shape[0] == self.num_states

    @property
    def num_states(self):
        return self.next_states.shape[0]

    @property
    def num_inputs(self):
        return self.next_states.shape[1]

    @staticmethod
    def from_next_states(next_states: tf.Tensor) -> "StateTransitionGraph":
        num_states = next_states.shape[0]
        num_inputs = next_states.shape[1]
        previous_states_accum: List[List[Tuple[int, int]]] = [
            [] for _ in range(num_states)
        ]
        for state in range(num_states):
            for input_sym in range(num_inputs):
                next_state = next_states[state, input_sym]
                previous_states_accum[next_state].append((state, input_sym))

        previous_states = tf.ragged.constant(previous_states_accum, inner_shape=(2,))

        return StateTransitionGraph(
            next_states=next_states, previous_states=previous_states
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, StateTransitionGraph):
            return tf.reduce_all(
                self.next_states == other.next_states
            ) and tf.reduce_all(self.previous_states == other.previous_states)
        else:
            return NotImplemented

    def unroll(self, num_steps: int):
        return UnrolledStateTransitionGraph(
            next_states=tf.tile(
                self.next_states[None],
                multiples=[num_steps] + [1] * int(tf.rank(self.next_states)),
            ),
            # Roundabout way to keep additional dimensions becoming ragged.
            previous_states=tf.RaggedTensor.from_uniform_row_length(
                tf.concat([self.previous_states] * num_steps, axis=0), self.num_states
            ),
        )


@dataclass
class UnrolledStateTransitionGraph:
    # Tensor of Time x |States| x |Inputs|
    next_states: tf.Tensor
    # RaggedTensor of Time x |States| x |PrevStates| x 2. Last dimension is pair previous state and transition input
    previous_states: tf.RaggedTensor

    def __post_init__(self):
        tf.debugging.assert_type(self.next_states, tf_type=tf.int32)
        tf.debugging.assert_rank(self.next_states, 3)
        assert self.previous_states.dtype == tf.int32
        assert self.previous_states.shape[1] == self.num_states
        assert self.previous_states.shape[0] == self.num_steps

    @property
    def num_states(self) -> int:
        return self.next_states.shape[1]

    @property
    def num_inputs(self) -> int:
        return self.next_states.shape[2]

    @property
    def num_steps(self) -> int:
        return self.next_states.shape[0]

    @staticmethod
    def from_next_states(next_states: tf.Tensor) -> "UnrolledStateTransitionGraph":
        num_steps = next_states.shape[0]
        num_states = next_states.shape[1]
        num_inputs = next_states.shape[2]
        previous_states_accum: List[List[Tuple[int, int]]] = [
            [] for _ in range(num_states * num_steps)
        ]
        for step in range(num_steps):
            for state in range(num_states):
                for input_sym in range(num_inputs):
                    next_state = next_states[step, state, input_sym]
                    previous_states_accum[step * num_states + next_state].append(
                        (state, input_sym)
                    )

        previous_states = tf.RaggedTensor.from_uniform_row_length(
            tf.ragged.constant(previous_states_accum, inner_shape=(2,)), num_states
        )
        return UnrolledStateTransitionGraph(
            next_states=next_states, previous_states=previous_states
        )

    @staticmethod
    def concat_unrolled_state_transitions(
        state_transitions: Sequence["UnrolledStateTransitionGraph"],
    ) -> "UnrolledStateTransitionGraph":
        next_states = tf.concat([st.next_states for st in state_transitions], axis=0)

        return UnrolledStateTransitionGraph.from_next_states(next_states=next_states)

    def concat(
        self, other: "UnrolledStateTransitionGraph"
    ) -> "UnrolledStateTransitionGraph":
        return self.concat_unrolled_state_transitions([self, other])

    def ensure_length(self, msg_length: int) -> tf.Tensor:
        return tf.ensure_shape(self.next_states, [msg_length, None, None])

    def __eq__(self, other: object) -> bool:
        if isinstance(other, UnrolledStateTransitionGraph):
            return tf.reduce_all(
                tf.equal(self.next_states, other.next_states)
            ) and tf.reduce_all(tf.equal(self.previous_states, other.previous_states))
        else:
            return NotImplemented


@dataclass
class Trellis:
    state_transitions: StateTransitionGraph
    output_table: tf.Tensor

    def __post_init__(self):
        tf.debugging.assert_type(self.output_table, tf_type=tf.float32)
        tf.debugging.assert_rank(self.output_table, 3)
        assert self.output_table.shape[0] == self.num_states
        assert self.output_table.shape[1] == self.num_inputs

    @property
    def num_outputs(self) -> int:
        return self.output_table.shape[2]

    @property
    def num_inputs(self) -> int:
        return self.state_transitions.num_inputs

    @property
    def num_states(self) -> int:
        return self.state_transitions.num_states

    @property
    def next_states(self) -> int:
        return self.state_transitions.next_states

    def concat(self, trellis2: "Trellis") -> "Trellis":
        if self.check_state_table_compatibility(trellis2):
            return Trellis(
                state_transitions=self.state_transitions,
                output_table=tf.concat(
                    [self.output_table, trellis2.output_table], axis=2
                ),
            )
        else:
            raise ValueError("Input trellis is not compatible with source trellis")

    def with_systematic(self) -> "Trellis":
        np_output_table = np.zeros(
            (self.output_table.shape[0], self.output_table.shape[1], 1)
        )
        np_output_table[:, 1] = 1
        id_output_table = tf.constant(np_output_table, dtype=self.output_table.dtype)
        return Trellis(self.state_transitions, id_output_table).concat(self)

    def check_state_table_compatibility(self, trellis2: "Trellis") -> bool:
        return self.state_transitions == trellis2.state_transitions

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Trellis):
            return self.state_transitions == other.state_transitions and tf.reduce_all(
                self.output_table == other.output_table
            )
        else:
            return NotImplemented

    def __mul__(self, other):
        return Trellis(self.state_transitions, self.output_table * other)

    def __add__(self, other):
        return Trellis(self.state_transitions, self.output_table + other)

    def __sub__(self, other):
        return Trellis(self.state_transitions, self.output_table - other)

    def __truediv__(self, other):
        return Trellis(self.state_transitions, self.output_table / other)

    def training(self) -> "TrainableTrellis":
        return TrainableTrellis(
            state_transitions=self.state_transitions,
            output_table=tf.Variable(self.output_table),
        )

    def validating(self) -> "Trellis":
        return self

    def unroll(self, num_steps: int) -> "UnrolledTrellis":
        return UnrolledTrellis(
            state_transitions=self.state_transitions.unroll(num_steps),
            output_tables=tf.tile(
                self.output_table[None],
                [num_steps] + [1] * int(tf.rank(self.output_table)),
            ),
        )


@dataclass
class TrainableTrellis(Trellis):
    state_transitions: StateTransitionGraph
    output_table: tf.Variable

    def training(self) -> "TrainableTrellis":
        return self

    def validating(self) -> "Trellis":
        return Trellis(
            state_transitions=self.state_transitions,
            output_table=self.output_table.value(),
        )


@dataclass
class UnrolledTrellis:
    state_transitions: UnrolledStateTransitionGraph
    # Time x States x Inputs x Outputs
    output_tables: tf.Tensor

    def __post_init__(self):
        tf.debugging.assert_type(self.output_tables, tf_type=tf.float32)
        tf.debugging.assert_rank(self.output_tables, 4)
        assert self.output_tables.shape[0] == self.num_steps
        assert self.output_tables.shape[1] == self.num_states
        assert self.output_tables.shape[2] == self.num_inputs

    @property
    def num_outputs(self) -> int:
        return self.output_tables.shape[3]

    @property
    def num_inputs(self) -> int:
        return self.state_transitions.num_inputs

    @property
    def num_states(self) -> int:
        return self.state_transitions.num_states

    @property
    def next_states(self) -> int:
        return self.state_transitions.next_states

    @property
    def num_steps(self) -> int:
        return self.state_transitions.num_steps

    @staticmethod
    def concat_unrolled_trellises(
        trellises: Sequence["UnrolledTrellis"],
    ):
        return UnrolledTrellis(
            state_transitions=UnrolledStateTransitionGraph.concat_unrolled_state_transitions(
                [trellis.state_transitions for trellis in trellises]
            ),
            output_tables=tf.concat(
                [trellis.output_tables for trellis in trellises], axis=0
            ),
        )

    def concat_time(self, other: "UnrolledTrellis"):
        return self.concat_unrolled_trellises([self, other])

    def check_state_table_compatibility(self, trellis2: "UnrolledTrellis") -> bool:
        return self.state_transitions == trellis2.state_transitions

    def concat_outputs(
        self, other: "UnrolledTrellis", check_compatibility=True
    ) -> "UnrolledTrellis":
        if (not check_compatibility) or self.check_state_table_compatibility(other):
            return UnrolledTrellis(
                state_transitions=self.state_transitions,
                output_tables=tf.concat(
                    [self.output_tables, other.output_tables], axis=3
                ),
            )
        else:
            raise ValueError("Input trellis is not compatible with source trellis")

    def with_systematic(self) -> "UnrolledTrellis":
        np_output_table = np.zeros((self.num_states, self.num_inputs, 1))
        np_output_table[:, 1] = 1
        id_output_table = tf.tile(
            tf.constant(np_output_table, dtype=self.output_tables.dtype)[None],
            [self.num_steps, 1, 1, 1],
        )
        return UnrolledTrellis(self.state_transitions, id_output_table).concat_outputs(
            self
        )

    def ensure_length(self, msg_length: int):
        output_tables = tf.ensure_shape(
            self.output_tables, [msg_length, None, None, None]
        )
        next_states = self.state_transitions.ensure_length(msg_length)
        return output_tables, next_states

    def training(self):
        TrainableUnrolledTrellis(
            state_transitions=self.state_transitions, output_tables=self.output_tables
        )

    def validating(self) -> "UnrolledTrellis":
        return self

    def __mul__(self, other: object) -> "UnrolledTrellis":
        return UnrolledTrellis(
            state_transitions=self.state_transitions,
            output_tables=self.output_tables * other,
        )

    def __add__(self, other: object) -> "UnrolledTrellis":
        return UnrolledTrellis(
            state_transitions=self.state_transitions,
            output_tables=self.output_tables + other,
        )

    def __sub__(self, other: object) -> "UnrolledTrellis":
        return UnrolledTrellis(
            state_transitions=self.state_transitions,
            output_tables=self.output_tables - other,
        )

    def __truediv__(self, other: object) -> "UnrolledTrellis":
        return UnrolledTrellis(
            state_transitions=self.state_transitions,
            output_tables=self.output_tables / other,
        )


class TrainableUnrolledTrellis(UnrolledTrellis):
    state_transitions: UnrolledStateTransitionGraph
    output_tables: tf.Variable

    def training(self) -> "TrainableUnrolledTrellis":
        return self

    def validating(self) -> "UnrolledTrellis":
        return UnrolledTrellis(
            state_transitions=self.state_transitions,
            output_tables=self.output_tables.value(),
        )
