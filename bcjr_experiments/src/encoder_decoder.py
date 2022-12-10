from typing import List, Mapping, Tuple
from dataclasses import dataclass

import tensorflow as tf

from .utils import create_subrecords
from .typing_utils import DataClass
from .decoders import SoftDecoder, SoftDecoderInput
from .channels import NoisyChannel
from .component import (
    Component,
    KnownChannelTensorComponent,
    KnownChannels,
    Record,
    RecordKey,
    ScalarRecord,
    TensorComponent,
    TensorRecord,
)
from .metric import bit_error_metrics, cross_entropy_with_logits


@dataclass
class EncoderDecoderSettings:
    encoder: DataClass
    constraint: DataClass
    channel: DataClass
    decoder: DataClass
    name: str
    rate: Tuple[int, int]
    num_input_channels: int
    num_output_channels: int


# Not a TensorComponent because `output` is not in the return records
class EncoderDecoder(Component[tf.Tensor], KnownChannels):
    name = "encoder_decoder"

    def __init__(
        self,
        encoder: KnownChannelTensorComponent,
        constraint: TensorComponent,
        channel: NoisyChannel,
        decoder: SoftDecoder,
        name=name,
        **kwargs,
    ) -> None:
        self.name = name
        self.encoder = encoder
        self.constraint = constraint
        self.channel = channel
        self.decoder = decoder

    @property
    def rate(self) -> Tuple[int, int]:
        return (self.encoder.num_input_channels, self.encoder.num_output_channels)

    @property
    def num_input_channels(self) -> int:
        return self.encoder.num_input_channels

    @property
    def num_output_channels(self) -> int:
        return self.decoder.num_output_channels

    def call(self, source_data: tf.Tensor) -> Mapping[str, Record]:
        """
        source_data : Batch x Time x Channels. Usually Channels=1
        """
        # Run the encoder and decoder
        code_records = self.encoder(source_data)
        constraint_records = self.constraint(code_records[RecordKey.OUTPUT].value)
        channel_records = self.channel(constraint_records[RecordKey.OUTPUT].value)
        received_symbols = channel_records[RecordKey.OUTPUT].value
        decoder_records = self.decoder(
            self.decoder.uniform_prior_input(received_symbols)
        )

        # Compute the metrics
        msg_confidence = decoder_records[RecordKey.OUTPUT].value
        bit_error_results = bit_error_metrics(
            original_msg=source_data, msg_confidence=msg_confidence
        )
        cross_entropy_result = cross_entropy_with_logits(
            original_msg=source_data, msg_confidence=msg_confidence
        )

        return {
            RecordKey.INPUT: TensorRecord(source_data),
            "ber": TensorRecord(bit_error_results["ber"], track_progress=True),
            "bler": TensorRecord(bit_error_results["bler"], track_progress=True),
            "cross_entropy": TensorRecord(cross_entropy_result, track_progress=True),
            RecordKey.LOSS: ScalarRecord(
                tf.reduce_mean(cross_entropy_result),
                track_progress=True,
            ),
            **create_subrecords(self.encoder.name, code_records),
            **create_subrecords(self.constraint.name, constraint_records),
            **create_subrecords(self.channel.name, channel_records),
            **create_subrecords(self.decoder.name, decoder_records),
        }

    def __call__(self, source_data: tf.Tensor) -> Mapping[str, Record]:
        """`source_data` = Batch x Time - A float tensor with binary values. Represents the data to transmit."""
        return self.call(source_data)

    def training(self):
        self.encoder.training()
        self.decoder.training()
        self.constraint.training()
        self.channel.training()

    def validating(self):
        self.encoder.validating()
        self.decoder.validating()
        self.constraint.validating()
        self.channel.validating()

    def parameters(self) -> List[tf.Variable]:
        return (
            self.encoder.parameters()
            + self.constraint.parameters()
            + self.channel.parameters()
            + self.decoder.parameters()
        )

    def settings(self):
        return EncoderDecoderSettings(
            encoder=self.encoder.settings(),
            constraint=self.constraint.settings(),
            channel=self.channel.settings(),
            decoder=self.decoder.settings(),
            name=self.name,
            rate=self.rate,
            num_input_channels=self.num_input_channels,
            num_output_channels=self.num_output_channels,
        )
