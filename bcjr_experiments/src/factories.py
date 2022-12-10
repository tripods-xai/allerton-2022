from typing import Callable, Mapping, Sequence
import tensorflow as tf

from .trellis import UnrolledTrellis
from .constants import (
    TURBOAE_EXACT_TABLE1_BITS_3_98,
    TURBOAE_EXACT_TABLE1_COMPOSITE,
    TURBOAE_EXACT_TABLE2_BITS_3_98,
    TURBOAE_EXACT_TABLE2_COMPOSITE,
    TURBOAE_APPROXIMATED_CODE1_BITS_3_98,
    TURBOAE_APPROXIMATED_CODE2_BITS_3_98,
)
from .engine import ValidatorCallback, StopWhenConfident
from .encoders import GeneralizedConvolutionalCode
from .interleaver import (
    FixedPermuteInterleaver,
    Interleaver,
    RandomPermuteInterleaver,
    TurboAEInterleaver,
)
from .datasets import BinaryMessages
from .decoders import BCJRDecoder, TurboDecoder, SoftDecoder, HazzysTurboDecoder
from .utils import create_unique_name
from .power_constraints import ScaleConstraint
from .component import IdentityComponent, KnownChannelTensorComponent, TensorComponent
from .channels import NoisyChannel, AWGN, AdditiveT
from .encoders import (
    TrellisCode,
    AffineConvolutionalCode,
    NonsystematicTurboEncoder,
    SystematicTurboEncoder,
    TurboEncoder,
)
from .encoder_decoder import EncoderDecoder


def _turboae_approximated_nonsys_window5(
    interleaver: Interleaver, block_len: int, delay: int, **kwargs
) -> TurboEncoder[TrellisCode, TrellisCode]:
    noninterleaved_encoder = AffineConvolutionalCode(
        TURBOAE_APPROXIMATED_CODE1_BITS_3_98.weights,
        TURBOAE_APPROXIMATED_CODE1_BITS_3_98.bias,
        num_steps=block_len,
        delay=delay,
    )
    interleaved_encoder = AffineConvolutionalCode(
        TURBOAE_APPROXIMATED_CODE2_BITS_3_98.weights,
        TURBOAE_APPROXIMATED_CODE2_BITS_3_98.bias,
        num_steps=block_len,
        delay=delay,
    )
    return NonsystematicTurboEncoder(
        noninterleaved_encoder=noninterleaved_encoder,
        interleaved_encoder=interleaved_encoder,
        interleaver=interleaver,
    )


def _turboae_exact_nonsys_nobd_window5(
    interleaver: Interleaver, block_len: int, delay: int, **kwargs
) -> TurboEncoder[TrellisCode, TrellisCode]:
    noninterleaved_encoder = GeneralizedConvolutionalCode(
        TURBOAE_EXACT_TABLE1_BITS_3_98, num_steps=block_len, delay=delay
    )
    interleaved_encoder = GeneralizedConvolutionalCode(
        TURBOAE_EXACT_TABLE2_BITS_3_98, num_steps=block_len, delay=delay
    )

    return NonsystematicTurboEncoder(
        noninterleaved_encoder=noninterleaved_encoder,
        interleaved_encoder=interleaved_encoder,
        interleaver=interleaver,
    )


def _turboae_exact_nonsys_bd_window5(
    interleaver: Interleaver, block_len: int, delay: int, **kwargs
) -> TurboEncoder[TrellisCode, TrellisCode]:
    # Shape is Inputs x Window x Channels

    # Build encoder 1 (noninterleaved)
    all_codes_1 = [
        GeneralizedConvolutionalCode(table=table, num_steps=steps, delay=delay,)
        for table, steps in TURBOAE_EXACT_TABLE1_COMPOSITE
    ]

    noninterleaved_encoder = TrellisCode(
        trellises=UnrolledTrellis.concat_unrolled_trellises(
            [c.trellises for c in all_codes_1]
        ),
        normalize_output_table=False,
        delay_state_transitions=all_codes_1[0].delay_state_transitions,
    )
    assert noninterleaved_encoder.num_steps == block_len

    # Build encoder 2 (interleaved)
    all_codes_2 = [
        GeneralizedConvolutionalCode(table=table, num_steps=steps, delay=delay,)
        for table, steps in TURBOAE_EXACT_TABLE2_COMPOSITE
    ]
    interleaved_encoder = TrellisCode(
        trellises=UnrolledTrellis.concat_unrolled_trellises(
            [c.trellises for c in all_codes_2]
        ),
        normalize_output_table=False,
        delay_state_transitions=all_codes_2[0].delay_state_transitions,
    )
    assert interleaved_encoder.num_steps == block_len
    print(
        f"Using delays {noninterleaved_encoder.delay} and {interleaved_encoder.delay}"
    )

    return NonsystematicTurboEncoder(
        noninterleaved_encoder=noninterleaved_encoder,
        interleaved_encoder=interleaved_encoder,
        interleaver=interleaver,
    )


def build_validator_callbacks(args_dict) -> Sequence[ValidatorCallback]:
    callbacks = []
    if args_dict["stop_when_confident"]:
        callbacks.append(StopWhenConfident(**args_dict))
    return callbacks


INTERLEAVER_FACTORIES: Mapping[str, Callable[..., Interleaver]] = {
    FixedPermuteInterleaver.name: FixedPermuteInterleaver,
    RandomPermuteInterleaver.name: RandomPermuteInterleaver,
    TurboAEInterleaver.name: TurboAEInterleaver,
}


def conv_75_0(block_len: int, **kwargs) -> AffineConvolutionalCode:
    return AffineConvolutionalCode(
        tf.constant([[1, 1, 1], [1, 0, 1]]), tf.constant([0, 0]), num_steps=block_len
    )


def conv_15_7(block_len: int, **kwargs) -> TrellisCode:
    return conv_75_0(block_len=block_len, **kwargs).to_rsc()


def turbo_155_7(
    interleaver: Interleaver, block_len: int, **kwargs
) -> SystematicTurboEncoder[TrellisCode, TrellisCode]:
    base = conv_75_0(block_len=block_len, **kwargs)
    interleaved_encoder = base.to_rc()
    noninterleaved_encoder = interleaved_encoder.with_systematic()
    return SystematicTurboEncoder(
        noninterleaved_encoder=noninterleaved_encoder,
        interleaved_encoder=interleaved_encoder,
        interleaver=interleaver,
    )


def turbo_lte(
    interleaver: Interleaver, block_len: int, **kwargs
) -> SystematicTurboEncoder[TrellisCode, TrellisCode]:
    base = AffineConvolutionalCode(
        tf.constant([[1, 0, 1, 1], [1, 1, 0, 1]]),
        tf.constant([0, 0]),
        num_steps=block_len,
    ).to_rc()
    noninterleaved_encoder = base.with_systematic()
    interleaved_encoder = base
    return SystematicTurboEncoder(
        noninterleaved_encoder=noninterleaved_encoder,
        interleaved_encoder=interleaved_encoder,
        interleaver=interleaver,
    )


def turboae_exact_nonsys_nobd_window5_delay0(
    interleaver: Interleaver, block_len: int, **kwargs
) -> TurboEncoder[TrellisCode, TrellisCode]:
    return _turboae_exact_nonsys_nobd_window5(
        interleaver, block_len=block_len, delay=0, **kwargs
    )


def turboae_exact_nonsys_nobd_window5_delay2(
    interleaver: Interleaver, block_len: int, **kwargs
) -> TurboEncoder[TrellisCode, TrellisCode]:
    return _turboae_exact_nonsys_nobd_window5(
        interleaver, block_len=block_len, delay=2, **kwargs
    )


def turboae_exact_nonsys_bd_window5_delay0(
    interleaver: Interleaver, block_len: int, **kwargs
):
    return _turboae_exact_nonsys_bd_window5(interleaver, block_len, delay=0, **kwargs)


def turboae_exact_nonsys_bd_window5_delay2(
    interleaver: Interleaver, block_len: int, **kwargs
):
    return _turboae_exact_nonsys_bd_window5(interleaver, block_len, delay=2, **kwargs)


def turboae_exact_rsc_nobd_window5(interleaver: Interleaver, block_len: int, **kwargs):
    table1 = tf.gather(TURBOAE_EXACT_TABLE1_BITS_3_98, indices=[1, 0], axis=1)
    noninterleaved_encoder = GeneralizedConvolutionalCode(
        table1, num_steps=block_len,
    ).to_rsc()
    table2 = tf.concat([table1[:, :1], TURBOAE_EXACT_TABLE2_BITS_3_98], axis=1)
    interleaved_encoder = GeneralizedConvolutionalCode(
        table2, num_steps=block_len
    ).to_rc()

    return SystematicTurboEncoder(
        noninterleaved_encoder=noninterleaved_encoder,
        interleaved_encoder=interleaved_encoder,
        interleaver=interleaver,
    )


def turboae_exact_rsc_bd_window5(
    interleaver: Interleaver, block_len: int, **kwargs
) -> TurboEncoder[TrellisCode, TrellisCode]:
    assert len(TURBOAE_EXACT_TABLE1_COMPOSITE) == len(TURBOAE_EXACT_TABLE2_COMPOSITE)
    assert all(
        steps1 == steps2
        for (_, steps1), (_, steps2) in zip(
            TURBOAE_EXACT_TABLE1_COMPOSITE, TURBOAE_EXACT_TABLE2_COMPOSITE
        )
    )
    # Shape is Inputs x Window x Channels

    # Build encoder 1 (noninterleaved)
    table1_composite = [
        (tf.gather(table, indices=[1, 0], axis=1), steps)
        for table, steps in TURBOAE_EXACT_TABLE1_COMPOSITE
    ]
    all_codes_1 = [
        GeneralizedConvolutionalCode(table=table, num_steps=steps,).to_rsc()
        for table, steps in table1_composite
    ]

    noninterleaved_encoder = TrellisCode(
        trellises=UnrolledTrellis.concat_unrolled_trellises(
            [c.trellises for c in all_codes_1]
        ),
        normalize_output_table=False,
        delay_state_transitions=all_codes_1[0].delay_state_transitions,
    )
    assert noninterleaved_encoder.num_steps == block_len

    # Build encoder 2 (interleaved)
    table2_composite = [
        (tf.concat([table1[:, :1], table2], axis=1), steps2)
        for (table1, steps1), (table2, steps2) in zip(
            table1_composite, TURBOAE_EXACT_TABLE2_COMPOSITE
        )
    ]

    all_codes_2 = [
        GeneralizedConvolutionalCode(table=table, num_steps=steps,).to_rc()
        for table, steps in table2_composite
    ]
    interleaved_encoder = TrellisCode(
        trellises=UnrolledTrellis.concat_unrolled_trellises(
            [c.trellises for c in all_codes_2]
        ),
        normalize_output_table=False,
        delay_state_transitions=all_codes_2[0].delay_state_transitions,
    )
    assert interleaved_encoder.num_steps == block_len
    print(
        f"Using delays {noninterleaved_encoder.delay} and {interleaved_encoder.delay}"
    )

    return SystematicTurboEncoder(
        noninterleaved_encoder=noninterleaved_encoder,
        interleaved_encoder=interleaved_encoder,
        interleaver=interleaver,
    )


def turboae_approximated_nonsys_nobd_window5_delay0(
    interleaver: Interleaver, block_len: int, **kwargs
):
    return _turboae_approximated_nonsys_window5(
        interleaver=interleaver, block_len=block_len, delay=0, **kwargs
    )


def turboae_approximated_nonsys_nobd_window5_delay2(
    interleaver: Interleaver, block_len: int, **kwargs
):
    return _turboae_approximated_nonsys_window5(
        interleaver=interleaver, block_len=block_len, delay=2, **kwargs
    )


def turboae_approximated_rsc_nobd_window5(
    interleaver: Interleaver, block_len: int, **kwargs
):
    weights1 = tf.gather(
        TURBOAE_APPROXIMATED_CODE1_BITS_3_98.weights, indices=[1, 0], axis=0
    )
    bias1 = tf.gather(TURBOAE_APPROXIMATED_CODE1_BITS_3_98.bias, indices=[1, 0], axis=0)
    noninterleaved_encoder = AffineConvolutionalCode(
        weights1, bias1, num_steps=block_len,
    ).to_rsc()
    weights2 = tf.concat(
        [weights1[:1], TURBOAE_APPROXIMATED_CODE2_BITS_3_98.weights,], axis=0,
    )
    bias2 = tf.concat([bias1[:1], TURBOAE_APPROXIMATED_CODE2_BITS_3_98.bias,], axis=0,)
    interleaved_encoder = AffineConvolutionalCode(
        weights2, bias2, num_steps=block_len,
    ).to_rc()
    return SystematicTurboEncoder(
        noninterleaved_encoder=noninterleaved_encoder,
        interleaved_encoder=interleaved_encoder,
        interleaver=interleaver,
    )


ENCODER_FACTORIES: Mapping[str, Callable[..., TensorComponent]] = {
    conv_15_7.__name__: conv_15_7,
    turbo_155_7.__name__: turbo_155_7,
    turbo_lte.__name__: turbo_lte,
    conv_75_0.__name__: conv_75_0,
    turboae_exact_nonsys_bd_window5_delay2.__name__: turboae_exact_nonsys_bd_window5_delay2,
    turboae_exact_nonsys_bd_window5_delay0.__name__: turboae_exact_nonsys_bd_window5_delay0,
    # turboae_approximated_nonsys_bd_window5_delay0.__name__: turboae_approximated_nonsys_bd_window5_delay0,
    # turboae_approximated_nonsys_bd_window5_delay0.__name__: turboae_approximated_nonsys_bd_window5_delay0,
    turboae_exact_nonsys_nobd_window5_delay2.__name__: turboae_exact_nonsys_nobd_window5_delay2,
    turboae_exact_nonsys_nobd_window5_delay0.__name__: turboae_exact_nonsys_nobd_window5_delay0,
    turboae_approximated_nonsys_nobd_window5_delay2.__name__: turboae_approximated_nonsys_nobd_window5_delay2,
    turboae_approximated_nonsys_nobd_window5_delay0.__name__: turboae_approximated_nonsys_nobd_window5_delay0,
    turboae_exact_rsc_bd_window5.__name__: turboae_exact_rsc_bd_window5,
    # turboae_approximated_rsc_bd_window5.__name__: turboae_approximated_rsc_bd_window5,
    turboae_exact_rsc_nobd_window5.__name__: turboae_exact_rsc_nobd_window5,
    turboae_approximated_rsc_nobd_window5.__name__: turboae_approximated_rsc_nobd_window5,
}
CHANNEL_FACTORIES: Mapping[str, Callable[..., NoisyChannel]] = {
    AWGN.name: AWGN,
    AdditiveT.name: AdditiveT,
    # TODO: Add the rest of them
}

# TODO: Other components should be changed to keep their default name in a class variable
CONSTRAINT_FACTORIES: Mapping[str, Callable[..., TensorComponent]] = {
    ScaleConstraint.name: ScaleConstraint,
    IdentityComponent.name: IdentityComponent,
}

DECODER_FACTORIES: Mapping[str, Callable[..., SoftDecoder]] = {
    BCJRDecoder.name: BCJRDecoder
}

DATASET_FACTORIES = {BinaryMessages.name: BinaryMessages}


def encoder_decoder_factory(
    encoder: str,
    constraint: str,
    channel: str,
    decoder: str,
    interleaver: str,
    block_len: int,
    **kwargs,
):
    channel = CHANNEL_FACTORIES[channel](**kwargs)
    constraint = CONSTRAINT_FACTORIES[constraint](**kwargs)

    interleaver = INTERLEAVER_FACTORIES[interleaver](block_len=block_len, **kwargs)
    encoder_decoder_name = encoder
    encoder = ENCODER_FACTORIES[encoder](
        interleaver=interleaver, block_len=block_len, **kwargs
    )
    if isinstance(encoder, TrellisCode):
        return trellis_encoder_decoder_factory(
            encoder=encoder,
            constraint=constraint,
            channel=channel,
            decoder=decoder,
            name=encoder_decoder_name,
            **kwargs,
        )
    elif isinstance(encoder, SystematicTurboEncoder):
        return systematic_turbo_encoder_decoder_factory(
            encoder=encoder,
            constraint=constraint,
            channel=channel,
            decoder=decoder,
            name=encoder_decoder_name,
            **kwargs,
        )
    elif isinstance(encoder, NonsystematicTurboEncoder):
        return nonsystematic_turbo_encoder_decoder_factory(
            encoder=encoder,
            constraint=constraint,
            channel=channel,
            decoder=decoder,
            name=encoder_decoder_name,
            **kwargs,
        )
    else:
        raise NotImplementedError()


def trellis_encoder_decoder_factory(
    encoder: TrellisCode,
    constraint: TensorComponent,
    channel: NoisyChannel,
    decoder: str,
    name: str,
    **kwargs,
):
    decoder = DECODER_FACTORIES[decoder](
        encoder=encoder, constraint=constraint, channel=channel, **kwargs
    )

    return EncoderDecoder(
        encoder,
        constraint=constraint,
        channel=channel,
        decoder=decoder,
        name=create_unique_name(name),
    )


def nonsystematic_turbo_encoder_decoder_factory(
    encoder: SystematicTurboEncoder[
        KnownChannelTensorComponent, KnownChannelTensorComponent
    ],
    constraint: TensorComponent,
    channel: NoisyChannel,
    decoder: str,
    name: str,
    **kwargs,
):
    decoder1 = DECODER_FACTORIES[decoder](
        encoder=encoder.noninterleaved_encoder,
        constraint=constraint,
        channel=channel,
        **kwargs,
    )
    decoder2 = DECODER_FACTORIES[decoder](
        encoder=encoder.interleaved_encoder,
        constraint=constraint,
        channel=channel,
        **kwargs,
    )
    decoder = TurboDecoder(
        decoder1, decoder2, interleaver=encoder.interleaver, **kwargs
    )

    return EncoderDecoder(
        encoder,
        constraint=constraint,
        channel=channel,
        decoder=decoder,
        name=create_unique_name(name),
    )


def systematic_turbo_encoder_decoder_factory(
    encoder: SystematicTurboEncoder[
        KnownChannelTensorComponent, KnownChannelTensorComponent
    ],
    constraint: TensorComponent,
    channel: NoisyChannel,
    decoder: str,
    name: str,
    **kwargs,
):
    decoder1 = DECODER_FACTORIES[decoder](
        encoder=encoder.noninterleaved_encoder,
        constraint=constraint,
        channel=channel,
        **kwargs,
    )
    decoder2 = DECODER_FACTORIES[decoder](
        encoder=encoder.interleaved_encoder.with_systematic(),
        constraint=constraint,
        channel=channel,
        **kwargs,
    )
    # decoder = SystematicTurboDecoder(
    #     decoder1, decoder2, interleaver=encoder.interleaver, **kwargs
    # )
    decoder = HazzysTurboDecoder(
        decoder1=decoder1,
        decoder2=decoder2,
        interleaver=encoder.interleaver,
        constraint=constraint,
        channel=channel,
        **kwargs,
    )

    return EncoderDecoder(
        encoder,
        constraint=constraint,
        channel=channel,
        decoder=decoder,
        name=create_unique_name(name),
    )
