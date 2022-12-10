import abc

from ..component import KnownChannelTensorComponent


class Encoder(KnownChannelTensorComponent):
    @property
    @abc.abstractmethod
    def delay(self) -> int:
        pass

    