from dataclasses import asdict, dataclass
from typing import (
    Any,
    Dict,
    Generic,
    Sequence,
    Mapping,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    TypedDict,
)
import abc

import tensorflow as tf

from .constants import RECORDS_SEP
from .typing_utils import DataClass


# Needs to be str because key can be arbitrary,
# otherwise would have made more sense to use an `Enum`.
# These are just commonly used key names.
class RecordKey:
    INPUT = "input"
    OUTPUT = "output"
    LOGIT_PRIOR = "logit_prior"
    LOGIT_INIT_PRIOR = "logit_init_prior"
    LOGIT_BITWISE_INIT_PRIOR = "logit_bitwise_init_prior"
    EXTRINSIC_POSTERIOR = "extrinsic_posterior"
    EXTRINSIC_INIT = "extrinsic_init"
    EXTRINSIC_BITWISE_INIT = "extrinsic_bitwise_init"
    INTERLEAVED_MESSAGE = "interleaved_message"
    NONINTERLEAVED_MESSAGE = "noninterleaved_message"
    LOSS = "loss"


# For the time being, I'm duplicating the information about
# tracking the data in the record between the two record
# types. Ideally I shouldn't have to do this, but I'm not
# sure how with dataclasses. This suggests dataclasses
# might not be the way to go...
# Using dataclasses to allow type checking when figuring out
# what to do with the record
@dataclass
class TensorRecord(tf.experimental.ExtensionType):
    value: tf.Tensor
    track_as_result: bool = True
    track_progress: bool = False
    track_tensorboard: bool = True
    prepared_for_progress: bool = False

    def tracking_settings(self):
        d = asdict(self)
        d.pop("value")
        return d


@dataclass
class ScalarRecord(tf.experimental.ExtensionType):
    # Allow tf.Tensor for scalar tensors. This is the way to go
    # until tensorflow sets up a proper set of type stubs
    value: Union[tf.Tensor, float]
    track_as_result: bool = True
    track_progress: bool = False
    track_tensorboard: bool = True
    prepared_for_progress: bool = False

    def tracking_settings(self):
        d = asdict(self)
        d.pop("value")
        return d


Record = Union[TensorRecord, ScalarRecord]

T = TypeVar("T")
Records = Mapping[str, Record]


class Component(Generic[T], metaclass=abc.ABCMeta):
    name: str

    @abc.abstractmethod
    def call(self, data: T) -> Records:
        pass

    def __call__(self, data: T) -> Records:
        # Leaving room for setup and teardown
        return self.call(data)

    @abc.abstractmethod
    def settings(self) -> DataClass:
        pass

    # These kind of training and validating methods will not work
    # I need to be careful about when tensorflow will retrace and
    # when it won't retrace and just used cached values.
    def training(self) -> None:
        pass

    def validating(self) -> None:
        pass

    def parameters(self) -> Sequence[tf.Variable]:
        return []


class TensorComponent(Component[tf.Tensor]):
    # Until extra keys are allowed in TypedDicts, I'll
    # use vague Mapping[str, Record]. I'll operate under the
    # assumption in my code that the dictionary always
    # contains an "input" and an "output" that are both TensorRecords.
    @abc.abstractmethod
    def call(self, data: tf.Tensor) -> Records:
        pass

    def __call__(self, data: tf.Tensor) -> Records:
        """Should also return an input and output record of tensor type."""
        return super().__call__(data)

    def concat(self, component2: "TensorComponent") -> "ConcatComponent":
        return ConcatComponent([self, component2])


@dataclass
class IdentityComponentSettings:
    name: str


class IdentityComponent(TensorComponent):
    name = "identity"

    def __init__(self, name=name, **kwargs) -> None:
        self.name = name

    def call(self, data: tf.Tensor) -> Records:
        return {
            RecordKey.INPUT: TensorRecord(data, track_tensorboard=False),
            RecordKey.OUTPUT: TensorRecord(data, track_tensorboard=False),
        }

    def settings(self) -> DataClass:
        return IdentityComponentSettings(name=self.name)


@dataclass
class ConcatComponentSettings:
    components: Sequence[Dict[str, Any]]


class ConcatComponent(TensorComponent):
    """Concatenates output of components along last dimension"""

    name: str = "concatenate"

    def __init__(self, components: Sequence[TensorComponent], name: str = name):
        assert len(components) > 0
        self.name = name
        self.components = components

    def call(self, data: tf.Tensor) -> Mapping[str, Record]:  # type: ignore[override]
        component_records = [component(data) for component in self.components]
        records = {
            RecordKey.INPUT: TensorRecord(data),
            RecordKey.OUTPUT: TensorRecord(
                tf.concat(
                    [record[RecordKey.OUTPUT] for record in component_records], axis=-1
                ),
            ),
            **{
                RECORDS_SEP.join([f"{component.name}_{i}", k]): v
                for i, (subrecords, component) in enumerate(
                    zip(component_records, self.components)
                )
                for k, v in subrecords.items()
            },
        }
        return records

    def settings(self):
        return ConcatComponentSettings(
            components=[component.settings() for component in self.components]
        )

    def training(self):
        for component in self.components:
            component.training()

    def validating(self):
        for component in self.components:
            component.validating()

    def parameters(self) -> Sequence[tf.Variable]:
        return [
            param for component in self.components for param in component.parameters()
        ]


class KnownChannels(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def num_input_channels(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def num_output_channels(self) -> int:
        pass

    @property
    def rate(self) -> Tuple[int, int]:
        return (self.num_input_channels, self.num_output_channels)


class KnownChannelTensorComponent(TensorComponent, KnownChannels):
    def with_systematic(self):
        return KnownChannelsConcatComponent(
            [KnownChannelIdentityComponent(self.num_input_channels), self]
        )


@dataclass
class KnownChannelIdentityComponentSettings:
    name: str
    num_input_channels: int


class KnownChannelIdentityComponent(IdentityComponent, KnownChannelTensorComponent):
    def __init__(
        self, num_input_channels: int, name=IdentityComponent.name, **kwargs
    ) -> None:
        self.name = name
        self._num_input_channels = num_input_channels

    @property
    def num_input_channels(self) -> int:
        return self._num_input_channels

    @property
    def num_output_channels(self) -> int:
        return self._num_input_channels

    def settings(self) -> DataClass:
        return KnownChannelIdentityComponentSettings(
            name=self.name, num_input_channels=self.num_input_channels
        )


class KnownChannelsConcatComponent(ConcatComponent, KnownChannelTensorComponent):
    def __init__(
        self,
        components: Sequence[KnownChannelTensorComponent],
        name=ConcatComponent.name,
    ):
        assert len(components) > 0
        assert all(
            components[0].num_input_channels == component.num_input_channels
            for component in components
        )
        self.name = name
        self.components: Sequence[KnownChannelTensorComponent] = components

    @property
    def num_input_channels(self) -> int:
        return self.components[0].num_input_channels

    @property
    def num_output_channels(self) -> int:
        return sum(component.num_output_channels for component in self.components)
