import abc
from typing import Dict, Mapping
from dataclasses import dataclass

import tensorflow as tf
import tensorflow_probability as tfp

from ..typing_utils import DataClass

from ..utils import sigma2snr, snr2sigma
from ..component import (
    Records,
    ScalarRecord,
    TensorRecord,
    RecordKey,
    TensorComponent,
    Record,
)


class NoisyChannel(TensorComponent):
    @abc.abstractmethod
    def noise_func(self, shape: tf.TensorShape, *args) -> tf.Tensor:
        """Generate iid noise of the given shape"""
        pass

    @abc.abstractmethod
    def log_likelihood(self, msg: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        """
        Compute the log likelihood of producing each output j given
        message bit i up to a constant dependent on the channel
        """
        pass

    @abc.abstractmethod
    def logit_posterior(self, msg: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        """
        Compute posterior logit (log-odds) of whether sent_i is output[0] or output[1] given received_i.

        Parameters
        ----------
        msg: Batch x Time x Channels
        outputs: Channels x 2  - outputs[c, 0] corresponds to output for input bit 0 and outputs[c, 1] for input bit 1

        Returns
        -------
        tf.Tensor: Batch x Time x Channels - value for time i channel c is log(P[sent_i=outputs[c, 1] | msg_i] / P[sent_i=outputs[c, 0] | msg_i])
        """
        pass

    @abc.abstractmethod
    def settings(self) -> DataClass:
        pass


class AdditiveNoisyChannel(NoisyChannel):
    def call(self, data: tf.Tensor) -> Mapping[str, Record]:
        """data = Batch x Time x Channel"""
        return {
            RecordKey.INPUT: TensorRecord(data),
            RecordKey.OUTPUT: TensorRecord(data + self.noise_func(tf.shape(data))),
        }


def broadcasted_msg_output(msg: tf.Tensor, outputs: tf.Tensor):
    """
    Parameters
    ----------
    - msg = Batch x Time x Channel - the message output from the channel
    - outputs = Time x ... x Channel - An array whose each entry is a Channel length output from the code.

    Returns
    -------
    - msg = Batch x Time x ...(1) x Channel - the message output from the channel
    - outputs = 1 x Time x ... x Channel - An array whose each entry is a Channel length output from the code.
    """
    msg = tf.ensure_shape(msg, shape=[None] * 3)
    msg_shape = tf.shape(msg)
    # We'll use broadcasting to compute for all pairs between msg and outputs
    expanded_msg_shape = tf.concat(
        [
            msg_shape[0:2],  # Batch x Time
            tf.ones(
                (tf.rank(outputs) - 2,), dtype=tf.int32
            ),  # Number of output dims excluding Time, Channels
            msg_shape[2:3],  # Channels
        ],
        axis=0,
    )
    msg = tf.reshape(msg, expanded_msg_shape)
    outputs = outputs[None]

    # B x K x 1 x 1 x ... x n and 1 x K x A0 x A1 x ... x n
    return msg, outputs


def gaussian_log_likelihood(msg: tf.Tensor, outputs: tf.Tensor, sigma) -> tf.Tensor:
    """
    Parameters
    ----------
    - msg = Batch x Time x Channel - the message output from the channel
    - outputs = Time x ... x Channel - An array whose each entry is a Channel length output from the code.

    Returns
    -------
    - chi_values = Batch x Time x ...
    """
    # Compute ln(Chi) values
    # chi_values[k, i, t] = log p(Y[k] | s[k] = i, s[k+1] = next_states[k, i, t])
    # B x K x 1 x 1 x ... x n - 1 x K x A0 x A1 x ... x n, result is B x K x A0 x A1 x ... x n
    msg, outputs = broadcasted_msg_output(msg, outputs)
    differences = msg - outputs

    # reduce on last axis, result is B x K x A0 x A1 x ...
    square_noise_sum = tf.math.reduce_sum(tf.square(differences), axis=-1)
    # return self._noise_constant - 1. / (2 * self.variance) * square_noise_sum
    return -1.0 / (2 * sigma**2) * square_noise_sum


def student_t_log_likelihood(
    msg: tf.Tensor,
    outputs: tf.Tensor,
    t_distribution: tfp.distributions.StudentT,
    sigma,
) -> tf.Tensor:
    """
    Parameters
    ----------
    - msg = Batch x Time x Channel - the message output from the channel
    - outputs = Time x ... x Channel - An array whose each entry is a Channel length output from the code.

    Returns
    -------
    - chi_values = Batch x Time x ...
    """

    # B x K x 1 x 1 x ... x n - 1 x K x A0 x A1 x ... x n, result is B x K x A0 x A1 x ... x n
    msg, outputs = broadcasted_msg_output(msg, outputs)
    differences = msg - outputs

    v = t_distribution.df
    rescaled_differences = differences / (sigma * tf.sqrt((v - 2) / v))
    # reduce on last axis, result is B x K x A0 x A1 x ...
    return tf.reduce_sum(t_distribution.log_prob(rescaled_differences), axis=-1)


@dataclass
class AWGNSettings:
    sigma: float
    snr: float
    name: str


class AWGN(AdditiveNoisyChannel):
    name: str = "awgn"

    def __init__(self, snr: float, name: str = name, **kwargs):
        self.name = name
        self.snr = snr
        self.sigma = snr2sigma(self.snr)

    def noise_func(self, shape):
        return self.sigma * tf.random.normal(shape)

    def log_likelihood(self, msg: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        return gaussian_log_likelihood(msg, outputs, self.sigma)

    def logit_posterior(self, msg: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        # B x T x C (x1) - (1x1x) C x 2 -> B x T x C x 2
        broadcasted_difference = tf.square(
            msg[..., None]
            - outputs[
                None,
                None,
            ]
        )

        # B x T x C - B x T x C  -> B x T x C
        return (-1 / (2 * self.sigma**2)) * (
            broadcasted_difference[..., 1] - broadcasted_difference[..., 0]
        )

    def call(self, data: tf.Tensor) -> Records:
        return {
            **super().call(data),
            "snr": ScalarRecord(self.snr),
            "sigma": ScalarRecord(self.sigma),
        }

    def settings(self) -> AWGNSettings:
        return AWGNSettings(self.sigma, self.snr, self.name)


@dataclass
class AdditiveTSettings:
    sigma: float
    v: float
    snr: float
    name: str


class AdditiveT(AdditiveNoisyChannel):
    name: str = "additive_t"

    def __init__(self, snr: float, v=3, name: str = name, **kwargs):
        self.name = name
        self.snr = snr
        self.v = v
        self.sigma = snr2sigma(self.snr)
        self.distribution = tfp.distributions.StudentT(df=self.v, loc=0, scale=1)

    def noise_func(self, shape):
        return (
            self.sigma
            * tf.sqrt((self.v - 2) / self.v)
            * self.distribution.sample(shape)
        )

    def log_likelihood(self, msg: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        return student_t_log_likelihood(msg, outputs, self.distribution, self.sigma)

    def logit_posterior(self, msg: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def settings(self) -> AdditiveTSettings:
        return AdditiveTSettings(self.sigma, self.v, self.snr, self.name)

    def call(self, data: tf.Tensor) -> Records:
        return {
            **super().call(data),
            "snr": ScalarRecord(self.snr),
            "sigma": ScalarRecord(self.sigma),
        }


class AdditiveTOnAWGN(AdditiveT, AWGN):
    name: str = "additive_t_on_awgn"

    def __init__(self, snr: float, v=3, name: str = name, **kwargs):
        super().__init__(snr=snr, v=v, name=name, **kwargs)

    def log_likelihood(self, msg: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        return super(AdditiveT, self).log_likelihood(msg, outputs)

    def logit_posterior(self, msg: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        return super(AdditiveT, self).logit_posterior(msg, outputs)

    def settings(self) -> AdditiveTSettings:  # type: ignore
        return super().settings()


class MultiplicativeNoisyChannel(NoisyChannel):
    def call(self, data: tf.Tensor) -> Mapping[str, Record]:
        result = data * self.noise_func(tf.shape(data))
        return {
            RecordKey.INPUT: TensorRecord(data),
            RecordKey.OUTPUT: TensorRecord(result),
        }


class BinarySymmetric(MultiplicativeNoisyChannel):
    def __init__(self, p_flip, name: str = "binary_symmetric", **kwargs) -> None:
        assert 0 < p_flip < 1
        self.p_flip = p_flip
        self.name = name

    def noise_func(self, shape):
        # Flipping a bit corresponds to multiplying by -1
        log_prob = tf.math.log([[1 - self.p_flip, self.p_flip]])
        num_elems = tf.math.reduce_prod(shape)
        return tf.cast(
            tf.random.categorical(log_prob, num_elems, dtype=tf.int32) * 2 - 1,
            dtype=tf.float32,
        )

    def log_likelihood(self, msg: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        """
        p(y | x) = 1-p if y = x else p
        chi_values[k, i, t] = log p(Y[k] | s[k] = i, s[k+1] = next_states[k, i, t])
                            = log [ Prod(s=1 -> n)[p(Ys[k] | outputs[k, i, t, s])] ]
                            = Sum(s=1 -> n) [ log p(Ys[k] | outputs[k, i, t, s]) ]
                            = Sum(s=1 -> n) [ log (1-p if Ys[k] = outputs[k, i, t, s] else p ]
        """
        msg, outputs = broadcasted_msg_output(msg, outputs)
        # B x K x 1 x 1 x ... x n == 1 x K x A0 x A1 x ... x n, result is B x K x A0 x A1 x ... x n
        # reduce last dim, results is B x K x A0 x A1 x ...
        return tf.reduce_sum(
            tf.math.log(tf.where(msg == outputs, 1 - self.p_flip, self.p_flip)), axis=-1
        )

    def logit_posterior(self, msg: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def call(self, data: tf.Tensor) -> Records:
        return {
            **super().call(data),
            "p_flip": ScalarRecord(self.p_flip),
        }


class BinaryErasure(MultiplicativeNoisyChannel):
    def __init__(
        self, p_flip, p_erase, name: str = "binary_symmetric", **kwargs
    ) -> None:
        # TODO
        raise NotImplementedError()
