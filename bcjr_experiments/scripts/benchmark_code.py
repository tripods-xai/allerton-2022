import argparse
from json import encoder
import logging

# This is to get it working inside the docker container
try:
    import src
except ModuleNotFoundError:
    import sys

    sys.path.append("/code/turbo-codes")
    import src

from src.factories import (
    CHANNEL_FACTORIES,
    CONSTRAINT_FACTORIES,
    DATASET_FACTORIES,
    DECODER_FACTORIES,
    ENCODER_FACTORIES,
    INTERLEAVER_FACTORIES,
    build_validator_callbacks,
    encoder_decoder_factory,
)
from src.interleaver import FixedPermuteInterleaver
from src.channels import AWGN
from src.power_constraints import ScaleConstraint
from src.decoders import BCJRDecoder
from src.datasets import binary_messages
from src.utils import (
    process_args_dict,
    get_stepped_range,
    create_unique_name,
    setup_logging,
)
from src.engine import Validator

parser = argparse.ArgumentParser()

encoder_group = parser.add_argument_group("Encoder")
encoder_group.add_argument(
    "encoder", choices=ENCODER_FACTORIES.keys(),
)


interleaver_group = parser.add_argument_group(
    "Interleaver", description="arguments for the interleaver"
)
interleaver_group.add_argument(
    "--interleaver",
    choices=INTERLEAVER_FACTORIES.keys(),
    default=FixedPermuteInterleaver.name,
)
interleaver_group.add_argument("--seed", type=int, default=2022)

turbo_group = parser.add_argument_group(
    "Turbo",
    description="arguments for the turbo code. Only relevant if using a turbo code.",
)
turbo_group.add_argument("--num_iter", type=int, required=False)

decoder_group = parser.add_argument_group(
    "Decoder", description="arguments for the decoder"
)
decoder_group.add_argument(
    "--decoder", choices=DECODER_FACTORIES.keys(), default=BCJRDecoder.name,
)
constraint_group = parser.add_argument_group(
    "Power Constraint", description="arguments for the power constraint"
)
constraint_group.add_argument(
    "--constraint", choices=CONSTRAINT_FACTORIES.keys(), default=ScaleConstraint.name,
)
channel_group = parser.add_argument_group(
    "Channel", description="arguments for the channel"
)
channel_group.add_argument(
    "--channel", choices=CHANNEL_FACTORIES.keys(), default=AWGN.name,
)
channel_group.add_argument(
    "--snr_start", type=float, default=-3.0,
)
channel_group.add_argument(
    "--snr_step_size", type=float, default=3.0,
)
channel_group.add_argument(
    "--snr_steps", type=int, default=2,
)
channel_group.add_argument(
    "--v", type=float, required=False, help="only used by t channel"
)

dataset_group = parser.add_argument_group(
    title="Dataset", description="arguments for the dataset"
)
dataset_group.add_argument(
    "--dataset", choices=DATASET_FACTORIES.keys(), default=binary_messages.__name__,
)
dataset_group.add_argument("--num_batches", type=int, default=1)
dataset_group.add_argument("--block_len", type=int, default=100)
dataset_group.add_argument("--batch_size", type=int, default=1000)

logging_group = parser.add_argument_group(
    "Logging", description="arguments for logging results"
)
logging_group.add_argument(
    "--write_tensorboard_dir",
    required=False,
    help="Where to write the tensorboard results; will not write tensorboard if not provided.",
)
logging_group.add_argument(
    "--write_json_dir",
    required=False,
    help="Where to write the json results; will not write json if not provided.",
)
logging_group.add_argument(
    "--write_plots_dir",
    required=False,
    help="Where to write the json results; will not write json if not provided.",
)
logging_group.add_argument(
    "--write_logfile",
    required=False,
    help="Where to write the logs; will only write to stdout if not provided",
)
logging_group.add_argument(
    "--no_tb_during_validation",
    action="store_true",
    help="Turns off tensorboard logging for stats while validating a component. This is within the experiment, not for each experiment",
)

confidence_callback_group = parser.add_argument_group(
    "Confidence Callback",
    description="arguments for the confidence callback that shuts validation when the confidence interval is small enough",
)
confidence_callback_group.add_argument("--stop_when_confident", action="store_true")
confidence_callback_group.add_argument("--patience", type=int, default=10)
confidence_callback_group.add_argument("--tolerance", type=float, default=1e-1)
confidence_callback_group.add_argument("--track", default="ber")

if __name__ == "__main__":
    args = parser.parse_args()
    experiment_id = create_unique_name(args.encoder)
    setup_logging(logging.DEBUG, experiment_id, logpath=args.write_logfile)

    args_dict = process_args_dict(vars(args))
    logging.info(args_dict)

    dataset = DATASET_FACTORIES[args_dict["dataset"]](**args_dict)

    snr_range = get_stepped_range(
        args_dict["snr_start"], args_dict["snr_steps"], args_dict["snr_step_size"]
    )
    logging.info(f"SNR Range {snr_range}")

    components = [encoder_decoder_factory(**args_dict, snr=snr) for snr in snr_range]

    experiment_tags = [
        {
            "snr": snr,
            "model_id": encoder_decoder.name,
            "channel": encoder_decoder.channel.name,
            "encoder": encoder_decoder.encoder.name,
            "decoder": encoder_decoder.decoder.name,
            "constraint": encoder_decoder.constraint.name,
            "dataset": dataset.name,
            "num_batches": dataset.cardinality,
            "block_len": args_dict["block_len"],
            "batch_size": args_dict["batch_size"],
            "v": args_dict.get("v"),
            "patience": args_dict.get("patience"),
            "tolerance": args_dict.get("tolerance"),
            "track": args_dict.get("track"),
            "interleaver": args_dict.get("interleaver"),
            "num_iter": args_dict.get("num_iter"),
        }
        for encoder_decoder, snr in zip(components, snr_range)
    ]

    callbacks = build_validator_callbacks(args_dict)

    validator = Validator(
        experiment_id=experiment_id,
        components=components,
        dataset=dataset,
        write_tensorboard_dir=args_dict.get("write_tensorboard_dir"),
        write_json_dir=args_dict.get("write_json_dir"),
        write_plots_dir=args_dict.get("write_plots_dir"),
        callbacks=callbacks,
        use_progress_bar=True,
        no_tb_during_validation=args_dict["no_tb_during_validation"],
        experiment_tags=experiment_tags,
    )
    validator.run()
