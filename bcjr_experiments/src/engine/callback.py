import abc
from dataclasses import dataclass
import logging
from typing import Callable, TypedDict

from ..typing_utils import DataClass
from ..component import Records
from ..utils import WithSettings
from ..constants import RECORDS_SEP


class CallbackResults(TypedDict):
    stop: bool
    records: Records


class ValidatorCallback(WithSettings):
    @abc.abstractmethod
    def __call__(self, records: Records) -> CallbackResults:
        pass

    @abc.abstractmethod
    def fresh_clone(self) -> "ValidatorCallback":
        pass


# TODO: The tolerane check should
# 1. check the other tracked metric and and check if the current std / sqrt(samples) < 10**(int(log10(check)) - 1)
# 2. Stop if patience is exceeded an tolerance check passes
@dataclass
class StopWhenConfidentSettings:
    track: str
    patience: int
    tolerance: float


class StopWhenConfident(ValidatorCallback):
    name: str = "stop_when_confident"

    def __init__(
        self,
        track: str,
        patience: int,
        tolerance: float = 0.1,
        name: str = name,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        - track: str - String that says which metric to track. It will
            use the mean and std of the metric
        - patience: int - Don't stop the validation until after this many steps.
        - tolerance: float - how much times the mean we want the confidence diameter to be.

        """
        self.num_calls = 0
        self.track = track
        self.patience = patience
        self.tolerance = tolerance

        self.name = name

    def __call__(self, records: Records) -> CallbackResults:
        self.num_calls += 1
        stop = False

        if self.num_calls > self.patience:
            mean_val = records[RECORDS_SEP.join([self.track, "mean"])].value
            std_confidence = records[
                RECORDS_SEP.join([self.track, "std_confidence"])
            ].value

            # Multiply by 2 to get diameter
            stop = 2 * std_confidence < self.tolerance * mean_val
            if stop:
                logging.info(
                    f"StopWhenConfidence reached its stop condition with std_confidence {std_confidence}, mean {mean_val}, and tolerance factor {self.tolerance}"
                )

        return CallbackResults(stop=stop, records=records)

    def settings(self) -> DataClass:
        return StopWhenConfidentSettings(
            track=self.track, patience=self.patience, tolerance=self.tolerance
        )

    def fresh_clone(self) -> ValidatorCallback:
        return StopWhenConfident(
            track=self.track,
            patience=self.patience,
            tolerance=self.tolerance,
            name=self.name,
        )
