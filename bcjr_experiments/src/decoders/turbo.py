from dataclasses import dataclass
from json import decoder
from typing import Any, Dict, Mapping
import tensorflow as tf

from ..channels import NoisyChannel
from ..metric import kl_divergence

from .decoder import (
    SoftDecoder,
    SoftDecoderInput,
    SoftDecoderWithDelay,
    compute_delay_logit_from_bitwise,
)
from ..component import Record, RecordKey, TensorRecord, TensorComponent
from .bcjr import BCJRDecoder, BCJRDecoderSettings
from ..interleaver import Interleaver


@dataclass
class TurboDecoderSettings:
    decoder1: BCJRDecoderSettings
    decoder2: BCJRDecoderSettings
    interleaver: Any
    num_iter: int
    num_noninterleaved_streams: int
    num_input_channels: int
    num_output_channels: int
    name: str


@dataclass
class TurboDecoderInput:
    received_symbols: tf.Tensor
    msg_noninterleaved: tf.Tensor
    msg_interleaved: tf.Tensor
    L_int_transmitted: tf.Tensor
    L_int_init: tf.Tensor
    known_information: tf.Tensor


class TurboDecoder(SoftDecoder):
    def __init__(
        self,
        decoder1: SoftDecoderWithDelay,
        decoder2: SoftDecoderWithDelay,
        interleaver: Interleaver,
        num_iter: int = 6,
        name="turbo_decoder",
        **kwargs,
    ):
        self.name = name
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.interleaver = interleaver
        self.num_iter = num_iter

        # decoder 1 tells us how much of the data is not interleaved
        self.num_noninterleaved_streams = self.decoder1.num_input_channels

        self.validate()

    @property
    def num_input_channels(self):
        return self.decoder1.num_input_channels + self.decoder2.num_input_channels

    @property
    def num_output_channels(self):
        return self.decoder1.num_output_channels

    def validate(self):
        if (
            self.decoder1.num_output_channels is not None
            and self.decoder2.num_output_channels is not None
        ):
            assert (
                self.decoder1.num_output_channels == self.decoder2.num_output_channels
            )

    def compute_known_information(self, received_symbols: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(received_symbols[..., :1])

    def prepare_turbo_decode_input(self, data: SoftDecoderInput) -> TurboDecoderInput:
        received_symbols, logit_prior, logit_init_prior = (
            data["received_symbols"],
            data["logit_prior"],
            data["logit_init_prior"],
        )
        msg_noninterleaved = received_symbols[:, :, : self.num_noninterleaved_streams]
        msg_interleaved = received_symbols[:, :, self.num_noninterleaved_streams :]

        return TurboDecoderInput(
            received_symbols=received_symbols,
            msg_noninterleaved=msg_noninterleaved,
            msg_interleaved=msg_interleaved,
            L_int_transmitted=logit_prior,
            L_int_init=logit_init_prior,
            known_information=self.compute_known_information(received_symbols),
        )

    def call(self, data: SoftDecoderInput) -> Mapping[str, Record]:
        """
        data has 2 entries:
        `logit_prior`: A 2d tensor of shape Batch x Time that holds the logit priors (log odds)
            that bit k is a 1
        `received_symbols`: A 3d tensor of shape Batch x Time x Channels that holds the corrupted mesage.

        Msg comes in with channels [Noninterleaved_1, ..., Noninterleaved_n, Interleaved_1,..., Interleaved_m]
        n = number of inputs to noninterleaved decoder (decoder1)
        m = number of inputs to interleaved decoder (decoder2)
        """
        turbo_input = self.prepare_turbo_decode_input(data)

        return {
            RecordKey.INPUT: TensorRecord(turbo_input.received_symbols),
            **self.decode(
                msg_noninterleaved=turbo_input.msg_noninterleaved,
                msg_interleaved=turbo_input.msg_interleaved,
                L_int_transmitted=turbo_input.L_int_transmitted,
                L_int_init=turbo_input.L_int_init,
                known_information=turbo_input.known_information,
            ),
        }

    def decode(
        self,
        msg_noninterleaved: tf.Tensor,
        msg_interleaved: tf.Tensor,
        L_int_transmitted: tf.Tensor,
        L_int_init: tf.Tensor,
        known_information: tf.Tensor,
    ) -> Mapping[str, Record]:
        """
        - msg_noninterleaved = Batch x Time x Channels
        - msg_interleaved = Batch x Time x Channels
        - L_int = Batch x Time-delay x 1 - the logit (log-odds) prior on whether source bit k is a 1
        - L_init_int = Batch x DelayInputs (2^delay) - This is p(u[0:d-1]). d := delay.
        - known_information = Batch x Time x 1 - Any additional log information that should be removed from iteration.


        TODO: For now we won't track the internals of the iterations, but this will
        be easy to add to the output dictionary if we want to.
        """
        batch_size = msg_interleaved.shape[0]

        L_int1_transmitted = L_int_transmitted
        L_int1_init = L_int_init
        L_int1 = tf.zeros(
            (batch_size, self.decoder1.delay + L_int_transmitted.shape[1], 1)
        )
        L_ext1_transmitted = tf.zeros_like(L_int1_transmitted)
        L_ext1_init_true = tf.zeros_like(L_int1_init)
        L_ext1_bitwise_init = tf.zeros((batch_size, self.decoder1.delay, 1))
        L_ext1 = tf.concat([L_ext1_bitwise_init, L_ext1_transmitted], axis=1)
        # L_ext1 = tf.zeros_like(L_int1)

        tracking_ext1_kl_d_array = tf.TensorArray(dtype=tf.float32, size=self.num_iter)
        tracking_ext2_kl_d_array = tf.TensorArray(dtype=tf.float32, size=self.num_iter)
        for i in tf.range(self.num_iter):

            decoder1_records = self.decoder1(
                SoftDecoderInput(
                    received_symbols=msg_noninterleaved,
                    logit_prior=L_int1_transmitted,
                    logit_init_prior=L_int1_init,
                )
            )
            L_ext1_transmitted, L_ext1_init_true, L_ext1_bitwise_init = (
                decoder1_records[RecordKey.EXTRINSIC_POSTERIOR].value,
                decoder1_records[RecordKey.EXTRINSIC_INIT].value,
                decoder1_records[RecordKey.EXTRINSIC_BITWISE_INIT].value,
            )
            L_ext1 = (
                tf.concat([L_ext1_bitwise_init, L_ext1_transmitted], axis=1)
                - known_information
            )

            # Measure how bad of an assumption independence on the L_ext1_init
            L_ext1_init_ind = compute_delay_logit_from_bitwise(L_ext1_bitwise_init)
            L_ext1_init_kl_d = kl_divergence(L_ext1_init_true, L_ext1_init_ind)
            tracking_ext1_kl_d_array = tracking_ext1_kl_d_array.write(
                i, L_ext1_init_kl_d
            )

            L_int2 = self.interleaver.interleave(L_ext1)
            L_int2_transmitted = L_int2[:, self.decoder2.delay :]
            L_int2_init = compute_delay_logit_from_bitwise(
                L_int2[:, : self.decoder2.delay]
            )

            decoder2_records = self.decoder2(
                SoftDecoderInput(
                    received_symbols=msg_interleaved,
                    logit_prior=L_int2_transmitted,
                    logit_init_prior=L_int2_init,
                )
            )
            L_ext2_transmitted, L_ext2_init_true, L_ext2_bitwise_init = (
                decoder2_records[RecordKey.EXTRINSIC_POSTERIOR].value,
                decoder2_records[RecordKey.EXTRINSIC_INIT].value,
                decoder2_records[RecordKey.EXTRINSIC_BITWISE_INIT].value,
            )
            L_ext2 = tf.concat([L_ext2_bitwise_init, L_ext2_transmitted], axis=1)

            # Measure how bad of an assumption independence on the L_ext2_init
            L_ext2_init_ind = compute_delay_logit_from_bitwise(L_ext2_bitwise_init)
            L_ext2_init_kl_d = kl_divergence(L_ext2_init_true, L_ext2_init_ind)
            tracking_ext2_kl_d_array = tracking_ext2_kl_d_array.write(
                i, L_ext2_init_kl_d
            )

            L_int1 = self.interleaver.deinterleave(L_ext2) - known_information
            L_int1_transmitted = L_int1[:, self.decoder1.delay :]
            L_int1_init = compute_delay_logit_from_bitwise(
                L_int1[:, : self.decoder1.delay]
            )

        L = L_ext1 + L_int1 + known_information
        return {
            RecordKey.NONINTERLEAVED_MESSAGE: TensorRecord(msg_noninterleaved),
            RecordKey.INTERLEAVED_MESSAGE: TensorRecord(msg_interleaved),
            RecordKey.LOGIT_PRIOR: TensorRecord(L_int_transmitted),
            RecordKey.LOGIT_INIT_PRIOR: TensorRecord(L_int_init),
            RecordKey.OUTPUT: TensorRecord(L),
            **{
                f"decoder1_init_kl_divergence_iter{i}": TensorRecord(
                    tracking_ext1_kl_d_array.read(i)
                )
                for i in range(self.num_iter)
            },
            **{
                f"decoder2_init_kl_divergence_iter{i}": TensorRecord(
                    tracking_ext2_kl_d_array.read(i)
                )
                for i in range(self.num_iter)
            },
        }

    def settings(self):
        return TurboDecoderSettings(
            decoder1=self.decoder1.settings(),
            decoder2=self.decoder2.settings(),
            interleaver=self.interleaver.settings(),
            num_iter=self.num_iter,
            num_noninterleaved_streams=self.num_noninterleaved_streams,
            num_input_channels=self.num_input_channels,
            num_output_channels=self.num_output_channels,
            name=self.name,
        )

    def training(self):
        self.decoder1.training()
        self.decoder2.training()

    def validating(self):
        self.decoder1.validating()
        self.decoder2.validating()

    def uniform_prior_input(self, received_symbols: tf.Tensor) -> SoftDecoderInput:
        return self.decoder1.uniform_prior_input(received_symbols)


class SystematicTurboDecoder(TurboDecoder):
    def __init__(
        self,
        decoder1: SoftDecoderWithDelay,
        decoder2: SoftDecoderWithDelay,
        interleaver: Interleaver,
        num_iter: int = 6,
        name="systematic_turbo_decoder",
        **kwargs,
    ):
        super().__init__(
            decoder1=decoder1,
            decoder2=decoder2,
            interleaver=interleaver,
            num_iter=num_iter,
            name=name,
        )

        assert (
            self.decoder1.delay == self.decoder1.delay == 0
        ), "Systematic Turbo Decoding does not support delay"

    @property
    def num_input_channels(self):
        # subtract 1 because we counted the sys channel twice, once for each
        # decoder.
        return self.decoder1.num_input_channels + self.decoder2.num_input_channels - 1

    def prepare_turbo_decode_input(self, data: SoftDecoderInput) -> TurboDecoderInput:
        received_symbols, logit_prior, logit_init_prior = (
            data["received_symbols"],
            data["logit_prior"],
            data["logit_init_prior"],
        )
        msg_noninterleaved = received_symbols[:, :, : self.num_noninterleaved_streams]
        msg_interleaved_no_sys = received_symbols[
            :, :, self.num_noninterleaved_streams :
        ]
        sys_interleaved = self.interleaver.interleave(received_symbols[:, :, :1])
        msg_interleaved = tf.concat([sys_interleaved, msg_interleaved_no_sys], axis=-1)

        return TurboDecoderInput(
            received_symbols=received_symbols,
            msg_noninterleaved=msg_noninterleaved,
            msg_interleaved=msg_interleaved,
            L_int_transmitted=logit_prior,
            L_int_init=logit_init_prior,
            known_information=self.compute_known_information(received_symbols),
        )


class HazzysTurboDecoder(SystematicTurboDecoder):
    def __init__(
        self,
        decoder1: BCJRDecoder,
        decoder2: BCJRDecoder,
        interleaver: Interleaver,
        constraint: TensorComponent,
        channel: NoisyChannel,
        num_iter: int = 6,
        name="hazzys_turbo_decoder",
        **kwargs,
    ):
        super().__init__(
            decoder1=decoder1,
            decoder2=decoder2,
            interleaver=interleaver,
            num_iter=num_iter,
            name=name,
        )

        self.channel = channel
        self.constraint = constraint

    def compute_known_information(self, received_symbols: tf.Tensor) -> tf.Tensor:
        """
        Hardcoded to assume
        - first channel is systematic
        - inputs are either 0 or 1

        All kinds of stuff could go wrong here.
        """
        outputs = self.constraint(tf.constant([[0.0, 1.0]]))[RecordKey.OUTPUT].value
        return self.channel.logit_posterior(received_symbols[..., :1], outputs)
