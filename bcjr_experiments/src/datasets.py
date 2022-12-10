from typing import Iterator, List, Literal, Protocol, Sequence, Tuple, Union
import tensorflow as tf


Cardinality = int
DataSampleValue = Union[tf.Tensor, Tuple[tf.Tensor, ...]]


class BatchedDataset(Protocol):
    @property
    def batch_size(self) -> int:
        ...

    @property
    def cardinality(self) -> Cardinality:
        ...

    def __iter__(self) -> Iterator["BatchedDataSample"]:
        ...


class BinaryMessages(BatchedDataset):
    name: str = "binary_messages"

    def __init__(
        self,
        batch_size: int,
        block_len: int,
        num_batches: int = None,
        num_channels: int = 1,
        name: str = name,
        **kwargs
    ) -> None:
        self._batch_size = batch_size
        self._block_len = block_len
        self._num_batches = num_batches
        self._num_channels = num_channels
        self.name = name

        self.dataset = binary_messages(batch_size, block_len, num_batches, num_channels)

    def __iter__(self) -> Iterator["BatchedDataSample"]:
        return iter(self.dataset)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def cardinality(self) -> Cardinality:
        return int(tf.data.experimental.cardinality(self.dataset))


class BatchedDataSample(tf.experimental.ExtensionType):
    value: DataSampleValue
    batch_size: int


def binary_messages(
    batch_size: int,
    block_len: int,
    num_batches: int = None,
    num_channels: int = 1,
) -> tf.data.Dataset:
    """
    Inputs
    ------
    batch_size: int
    block_len: int
    num_batches: int - If this is set to None, then the dataset will be of infinite
        size.

    Returns
    -------
    A tf.data.Dataset that yields random binary sequences of the specified size. Note
    that even if the num_batches is finite, iterating over the dataset again will
    produce different binary sequences.
    """
    data_shape = (batch_size, block_len, num_channels)
    dtype = tf.float32

    def data_gen() -> Iterator[BatchedDataSample]:
        while True:
            yield BatchedDataSample(
                value=tf.cast(
                    tf.random.uniform(data_shape, minval=0, maxval=2, dtype=tf.int32),
                    dtype=dtype,
                ),
                batch_size=batch_size,
            )

    dataset = tf.data.Dataset.from_generator(
        data_gen,
        output_signature=BatchedDataSample.Spec(
            value=tf.TensorSpec(shape=data_shape, dtype=tf.float32),
            batch_size=batch_size,
        ),
    )
    dataset = dataset.apply(
        tf.data.experimental.assert_cardinality(tf.data.INFINITE_CARDINALITY)
    )
    if num_batches is not None:
        dataset = dataset.take(num_batches)

    return dataset
