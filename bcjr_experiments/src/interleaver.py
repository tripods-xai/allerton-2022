import abc
from dataclasses import dataclass
from typing import Mapping, Optional

import tensorflow as tf

from .constants import TURBOAE_INTERLEAVER_PERMUTATION
from .component import Record, RecordKey, TensorComponent, TensorRecord


class Interleaver(TensorComponent):
    # TODO: If I want to log information from here, I will need to change the output
    # to a dictionary of records
    @abc.abstractmethod
    def deinterleave(self, msg: tf.Tensor) -> tf.Tensor:
        pass

    @abc.abstractmethod
    def interleave(self, msg: tf.Tensor) -> tf.Tensor:
        pass


@dataclass
class FixedPermuteInterleaverSettings:
    block_len: int
    permutation: tf.Tensor
    depermutation: tf.Tensor
    seed: Optional[int]
    name: str


class FixedPermuteInterleaver(Interleaver):
    name: str = "fixed_permute_interleaver"

    def __init__(
        self,
        block_len: int,
        permutation=None,
        depermutation=None,
        seed: int = None,
        name: str = name,
        **kwargs
    ):
        self.name = name
        self.block_len = block_len
        self.seed = seed
        if permutation is None:
            self.permutation = tf.random.shuffle(tf.range(block_len), seed=seed)
        else:
            self.permutation = permutation
        if depermutation is None:
            self.depermutation = tf.math.invert_permutation(self.permutation)
        else:
            self.depermutation = permutation

        self.validate()

    def validate(self):
        assert len(self.permutation) == self.block_len == len(self.depermutation)
        assert tf.reduce_all(
            tf.gather(self.permutation, self.depermutation) == tf.range(self.block_len)
        )

    @property
    def num_input_channels(self):
        return None

    @property
    def num_output_channels(self):
        return None

    def __len__(self):
        return self.block_len

    def call(self, data: tf.Tensor) -> Mapping[str, Record]:
        """
        `data` is a Nd tensor with shape Batch x Time x ...
        We interleave over the time dimension
        """
        return {
            RecordKey.INPUT: TensorRecord(data),
            RecordKey.OUTPUT: TensorRecord(self.interleave(data)),
        }

    def deinterleave(self, msg: tf.Tensor) -> tf.Tensor:
        return tf.gather(msg, self.depermutation, axis=1)

    def interleave(self, msg: tf.Tensor) -> tf.Tensor:
        return tf.gather(msg, self.permutation, axis=1)

    def settings(self) -> FixedPermuteInterleaverSettings:
        return FixedPermuteInterleaverSettings(
            permutation=self.permutation,
            depermutation=self.depermutation,
            block_len=self.block_len,
            seed=self.seed,
            name=self.name,
        )


class TurboAEInterleaver(FixedPermuteInterleaver):
    name = "turbo_ae_interleaver"

    def __init__(
        self,
        name: str = name,
        *,
        block_len=None,
        permutation=None,
        depermutation=None,
        seed=None,
        **kwargs
    ):
        interleaver_permutation = TURBOAE_INTERLEAVER_PERMUTATION
        super().__init__(
            block_len=len(interleaver_permutation),
            permutation=interleaver_permutation,
            depermutation=None,
            seed=None,
            name=name,
            **kwargs
        )


@dataclass
class RandomPermuteInterleaverSettings:
    block_len: int
    seed: Optional[int]
    name: str


class RandomPermuteInterleaver(Interleaver):
    name: str = "random_permute_interleaver"

    def __init__(self, block_len: int, seed: int = None, name: str = name, **kwargs):
        self.name = name
        self.block_len = block_len
        self.seed = seed
        if self.seed is not None:
            self.generator = tf.random.Generator.from_seed(seed)
        else:
            self.generator = tf.random.Generator.from_non_deterministic_state()
        # Put the underscore to mark that these should not be accessed.
        # They are ephemeral.
        self._permutation = None
        self._depermutation = None

    @property
    def num_input_channels(self):
        return None

    @property
    def num_output_channels(self):
        return None

    def __len__(self):
        return self.block_len

    def get_new_seeds(self, num_seeds=1):
        return self.generator.uniform(
            shape=[num_seeds, 2], minval=0, maxval=99999, dtype=tf.int64
        )

    def generate_permutations(self, batch_size):
        ta_perm = tf.TensorArray(
            tf.int32,
            size=batch_size,
            clear_after_read=True,
            element_shape=tf.TensorShape([self.block_len]),
        )
        ta_deperm = tf.TensorArray(
            tf.int32,
            size=batch_size,
            clear_after_read=True,
            element_shape=tf.TensorShape([self.block_len]),
        )
        seeds = self.get_new_seeds(num_seeds=batch_size)
        for i in tf.range(batch_size):
            permutation = tf.random.experimental.index_shuffle(
                tf.range(self.block_len), seed=seeds[i, :], max_index=self.block_len - 1
            )
            ta_perm = ta_perm.write(i, permutation)
            ta_deperm = ta_deperm.write(i, tf.math.invert_permutation(permutation))
        return ta_perm.stack(), ta_deperm.stack()

    def call(self, data: tf.Tensor) -> Mapping[str, Record]:
        """
        `data` is a Nd tensor with shape Batch x Time x ...
        We interleave over the time dimension.
        `call` will generate new permutations. If we need to repeat
        the interleaving or deinterleave, we should use the `interleave`
        or `deinterleave` methods.
        """
        batch_size = tf.shape(data)[0]
        self._permutation, self._depermutation = self.generate_permutations(batch_size)
        return {
            RecordKey.INPUT: TensorRecord(data),
            RecordKey.OUTPUT: TensorRecord(self.interleave(data)),
        }

    def deinterleave(self, msg: tf.Tensor) -> tf.Tensor:
        return tf.gather(msg, self._depermutation, axis=1, batch_dims=1)

    def interleave(self, msg: tf.Tensor) -> tf.Tensor:
        return tf.gather(msg, self._permutation, axis=1, batch_dims=1)

    def settings(self) -> RandomPermuteInterleaverSettings:
        return RandomPermuteInterleaverSettings(
            block_len=self.block_len, seed=self.seed, name=self.name
        )
