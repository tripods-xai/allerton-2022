from dataclasses import dataclass
import abc
from typing import Generic, List, TypeVar

import tensorflow as tf

from ..utils import create_subrecords
from ..typing_utils import DataClass
from ..component import (
    Records,
    KnownChannelTensorComponent,
    RecordKey,
    TensorRecord,
)
from ..interleaver import Interleaver
from .encoder import Encoder


@dataclass
class TurboEncoderSettings:
    noninterleaved_encoder: DataClass
    interleaved_encder: DataClass
    interleaver: DataClass
    systematic: bool
    name: str


T = TypeVar("T", bound=Encoder)
S = TypeVar("S", bound=Encoder)


class TurboEncoder(KnownChannelTensorComponent, Generic[T, S]):
    name: str = "turbo_encoder"

    def __init__(
        self,
        noninterleaved_encoder: T,
        interleaved_encoder: S,
        interleaver: Interleaver,
        name: str = name,
    ):
        self.noninterleaved_encoder = noninterleaved_encoder
        self.interleaved_encoder = interleaved_encoder
        self.interleaver = interleaver
        self.name = name

        self.validate()

    def validate(self):
        pass

    def call(self, data: tf.Tensor) -> Records:
        noninterleaved_encoded_records = self.noninterleaved_encoder(data)
        interleaved_msg_records = self.interleaver(data)
        interleaved_encoded_records = self.interleaved_encoder(
            interleaved_msg_records[RecordKey.OUTPUT].value
        )

        encoded = tf.concat(
            [
                noninterleaved_encoded_records[RecordKey.OUTPUT].value,
                interleaved_encoded_records[RecordKey.OUTPUT].value,
            ],
            axis=-1,
        )
        return {
            RecordKey.INPUT: TensorRecord(data),
            RecordKey.OUTPUT: TensorRecord(encoded),
            **create_subrecords(
                f"{self.noninterleaved_encoder.name}_noninterleaved",
                noninterleaved_encoded_records,
            ),
            **create_subrecords(f"{self.interleaver.name}", interleaved_msg_records),
            **create_subrecords(
                f"{self.interleaved_encoder.name}_interleaved",
                interleaved_encoded_records,
            ),
        }

    @property
    def num_input_channels(self):
        return self.noninterleaved_encoder.num_input_channels

    @property
    def num_output_channels(self):
        return (
            self.noninterleaved_encoder.num_output_channels
            + self.interleaved_encoder.num_output_channels
        )

    def settings(self) -> TurboEncoderSettings:
        return TurboEncoderSettings(
            noninterleaved_encoder=self.noninterleaved_encoder.settings(),
            interleaved_encder=self.interleaved_encoder.settings(),
            interleaver=self.interleaver.settings(),
            systematic=self.systematic,
            name=self.name,
        )

    def parameters(self) -> List[tf.Variable]:
        return (
            self.noninterleaved_encoder.parameters()
            + self.interleaved_encoder.parameters()
            + self.interleaver.parameters()
        )

    def training(self) -> None:
        self.noninterleaved_encoder.training()
        self.interleaved_encoder.training()
        self.interleaver.training()

    def validating(self) -> None:
        self.noninterleaved_encoder.validating()
        self.interleaved_encoder.validating()
        self.interleaver.validating()

    @property
    @abc.abstractmethod
    def systematic(self) -> bool:
        pass


class SystematicTurboEncoder(TurboEncoder[T, S]):
    @property
    def systematic(self) -> bool:
        return True


class NonsystematicTurboEncoder(TurboEncoder[T, S]):
    @property
    def systematic(self) -> bool:
        return False
