import math
from typing import Protocol, Dict, Any
from collections import defaultdict

from tqdm import tqdm
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

from ..constants import RECORDS_SEP
from ..typing_utils import DataClass
from ..component import Records, TensorRecord, ScalarRecord


class ResultsTracker(Protocol):
    def update(self, records: Records, steps: int = 1) -> "ResultsTracker":
        ...

    def results(self) -> Records:
        ...


class _NoopResultsTracker(ResultsTracker):
    def update(self, records: Records, steps: int = 1) -> "ResultsTracker":
        pass

    def results(self) -> Records:
        return {}


NoopResultsTracker = _NoopResultsTracker()


def records_for_logging(records: Records):
    return {
        name: float(record.value)
        for name, record in records.items()
        if isinstance(record, ScalarRecord) and record.track_progress
    }


class StatsResultsTracker(ResultsTracker):
    running_var_init = tfp.experimental.stats.RunningVariance.from_shape(shape=())

    def __init__(self) -> None:
        self.running_vars: Dict[str, Any] = defaultdict(
            lambda: StatsResultsTracker.running_var_init
        )
        self.tracking_settings: Dict[str, Dict[str, Any]] = {}
        self.n_steps = 0

    def update(self, records: Records, steps=1) -> Records:
        for name, record in records.items():
            if record.track_as_result:
                if isinstance(record, TensorRecord):
                    self.running_vars[name] = self.running_vars[name].update(
                        tf.reshape(record.value, (-1)), axis=0
                    )

                elif isinstance(record, ScalarRecord):
                    self.running_vars[name] = self.running_vars[name].update(
                        tf.constant(record.value), axis=None
                    )

                self.tracking_settings[name] = record.tracking_settings()

        self.n_steps += steps

        return {**records, **self.results()}

    def results(self) -> Records:
        output_dict = {
            **{
                RECORDS_SEP.join([name, "mean"]): ScalarRecord(
                    float(running_var.mean),
                    **{**self.tracking_settings[name], "prepared_for_progress": True}
                )
                for name, running_var in self.running_vars.items()
            },
            # Note that this std value will have slight bias. At large sample size though, this shouldn't be an issue.
            **{
                RECORDS_SEP.join([name, "std"]): ScalarRecord(
                    float(tf.math.sqrt(running_var.variance(ddof=1))),
                    **{
                        **self.tracking_settings[name],
                        "prepared_for_progress": True,
                        "track_progress": False,
                    }
                )
                for name, running_var in self.running_vars.items()
            },
            "steps": ScalarRecord(self.n_steps, track_tensorboard=False),
        }

        output_dict = {
            **output_dict,
            **{
                RECORDS_SEP.join([name, "std_confidence"]): ScalarRecord(
                    output_dict[RECORDS_SEP.join([name, "std"])].value
                    / math.sqrt(self.n_steps),
                    **{**self.tracking_settings[name], "prepared_for_progress": True}
                )
                for name in self.running_vars.keys()
            },
        }

        return output_dict


class ProgressBar(ResultsTracker):
    def close(self):
        ...


class _NoopProgressBar(ProgressBar, _NoopResultsTracker):
    def close(self):
        pass


NoopProgressBar = _NoopProgressBar()


class TqdmProgressBar(ProgressBar):
    def __init__(self, total: int = None) -> None:
        self.progress_bar = tqdm(total=total)
        self.total = total

    def update(self, records: Records, steps=1) -> "Records":
        results = {}
        for name, record in records.items():
            if record.prepared_for_progress and record.track_progress:
                if isinstance(record, TensorRecord):
                    results[name] = float(tf.reduce_mean(record.value))
                if isinstance(record, ScalarRecord):
                    results[name] = float(record.value)

        self.progress_bar.update(n=steps)
        self.progress_bar.set_postfix(**results)

        return records

    def results(self):
        raise NotImplementedError()  # Gross, this needs more work

    def close(self):
        self.progress_bar.close()
