from dataclasses import dataclass
from typing import Mapping

from .typing_utils import DataClass
from .component import TensorComponent, RecordKey, Record, TensorRecord


@dataclass
class ScaleConstraintSettings:
    scale: float
    bias: float
    name: str


class ScaleConstraint(TensorComponent):
    name = "scale_constraint"

    def __init__(self, scale=2.0, bias=-1.0, name=name, **kwargs) -> None:
        self.scale = scale
        self.bias = bias
        self.name = name

    def call(self, data) -> Mapping[str, Record]:
        result = self.scale * data + self.bias
        return {
            RecordKey.INPUT: TensorRecord(data),
            RecordKey.OUTPUT: TensorRecord(result),
        }

    def settings(self) -> DataClass:
        return ScaleConstraintSettings(scale=self.scale, bias=self.bias, name=self.name)
