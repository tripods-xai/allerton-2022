import abc
from datetime import datetime
import logging
import math
import os
import sys

import shortuuid
import numpy as np
import tensorflow as tf

from .typing_utils import DataClass
from .constants import ALPHABET, RECORDS_SEP

SU = shortuuid.ShortUUID(alphabet=ALPHABET)


class WithSettings(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def settings(self) -> DataClass:
        pass

    def __str__(self) -> str:
        return str(self.settings())


def dec2bitarray(arr, num_bits: int, little_endian: bool = False):
    if little_endian:
        shift_arr = tf.range(num_bits)
    else:
        shift_arr = tf.range(num_bits)[::-1]
    return tf.bitwise.right_shift(tf.expand_dims(arr, -1), shift_arr) % 2


def enumerate_binary_inputs(window: int) -> tf.Tensor:
    """Returns tensor of dimension 2 of all 2^n binary sequences of length `window`"""
    return dec2bitarray(tf.range(2**window), window)


def check_int(x: float) -> int:
    assert x.is_integer()
    return int(x)


def base_2_accumulator(length: int, little_endian: bool = False) -> tf.Tensor:
    powers_of_2 = tf.bitwise.left_shift(1, tf.range(length))
    if little_endian:
        return powers_of_2
    else:
        return powers_of_2[::-1]


def bitarray2dec(arr, little_endian=False, axis=-1) -> tf.Tensor:
    base_2 = base_2_accumulator(arr.shape[axis], little_endian=little_endian)
    return tf.tensordot(arr, base_2, axes=[[axis], [0]])


def sigma2snr(sigma):
    return -10 * math.log(sigma**2, 10)


def snr2sigma(snr):
    return math.sqrt(10 ** (-snr / 10))


def get_timestamp():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def create_unique_name(name_base):
    return f"{name_base}.{get_timestamp()}.{SU.random(length=4)}"


def create_subrecords(parent: str, code_records):
    return {RECORDS_SEP.join([f"{parent}", k]): v for k, v in code_records.items()}


# File utils
def safe_open_dir(dirpath: str) -> str:
    if not os.path.isdir(dirpath):
        print(f"Directory {dirpath} does not exist, creating it")
        os.makedirs(dirpath)
    return dirpath


def safe_create_file(filepath: str) -> str:
    dirpath = os.path.dirname(filepath)
    dirpath = safe_open_dir(dirpath)
    return filepath


# arg parse utils
def process_args_dict(args_dict):
    # If a value is None, that means it was not provided.
    # In that case, we want to fallback to the original default value
    # on the relevant object's constructor, so we remove all None
    # entries in the dict.
    args_dict = {k: v for k, v in args_dict.items() if v is not None}
    return args_dict


def get_stepped_range(start, steps, step_size):
    end = start + steps * step_size
    return np.linspace(start, end, num=steps + 1)


# Logging
def setup_logging(logging_level, experiment_id: str, logpath=None):
    logger = logging.getLogger()
    if len(logger.handlers) == 0:
        logger.setLevel(logging_level)
        formatter = logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s")
        if logpath is not None:
            fh = logging.FileHandler(
                os.path.join(safe_open_dir(logpath), f"{experiment_id}.txt")
            )
            fh.setLevel(logging_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Override the except hook to also log the exception
        def handle_exception(exc_type, exc_value, exc_traceback):
            logger.error(
                "Uncaught Exception", exc_info=(exc_type, exc_value, exc_traceback)
            )
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        sys.excepthook = handle_exception
    return logger
