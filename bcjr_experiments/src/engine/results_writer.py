from dataclasses import asdict, is_dataclass
import json
import logging
from typing import Any, Dict, List, Optional, Protocol, TypedDict

import pandas as pd
import numpy as np
import tensorflow as tf

from ..component import Component, Record, Records, ScalarRecord, TensorRecord


class TurboCodesJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        if tf.is_tensor(o):
            return o.numpy().tolist()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)


class ResultsWriter(Protocol):
    def add_result(
        self,
        records: Records,
        component: Component,
        experiment_settings: Dict[str, Any] = None,
    ) -> None:
        ...

    def flush(self) -> None:
        ...


class JSONResultsWriter(ResultsWriter):
    def __init__(self, path: str, write_every: int = 1) -> None:
        self.path = path
        self.write_every = write_every
        self.updates_since_write = 0
        self.entries: List[Dict[str, Any]] = []

        logging.info(f"Creating JSON writer at path {self.path}")

    def add_result(
        self,
        records: Records,
        component: Component,
        experiment_settings: Dict[str, Any] = None,
    ) -> None:
        new_entry = {
            "component": component.settings(),
            "experiment_settings": experiment_settings,
            "metrics": {},
        }
        for name, record in records.items():
            if record.track_as_result:
                if isinstance(record, ScalarRecord):
                    new_entry["metrics"][name] = record.value

        self.entries.append(new_entry)

        self.updates_since_write = (self.updates_since_write + 1) % self.write_every
        if self.updates_since_write == 0:
            self.flush()

    def flush(self):
        logging.debug("Flushing JSON results writer")
        from pprint import pprint

        with open(self.path, "w") as f:
            json.dump(self.entries, f, cls=TurboCodesJSONEncoder, indent=2)


class PlotterResultsWriter(ResultsWriter):
    def __init__(self, path: str, write_every: int = 1) -> None:
        self.path = path
        self.write_every = write_every
        self.updates_since_write = 0
        self.entries: List[Dict[str, Any]] = []

        logging.info(f"Creating Plotting writer at path {self.path}")

    def add_result(
        self,
        records: Records,
        component: Component,
        experiment_settings: Dict[str, Any] = None,
    ) -> None:
        if experiment_settings is None:
            experiment_settings = {}
        new_entry = {
            "model_id": component.name,
            **experiment_settings,
        }
        for name, record in records.items():
            if record.track_as_result:
                if isinstance(record, ScalarRecord):
                    new_entry[name] = record.value

        self.entries.append(new_entry)

        self.updates_since_write = (self.updates_since_write + 1) % self.write_every
        if self.updates_since_write == 0:
            self.flush()

    def flush(self):
        logging.debug("Flushing Plotter results writer")
        df = pd.DataFrame(self.entries)
        df.to_csv(self.path, index=False)


class _NoopResultsWriter(ResultsWriter):
    def add_result(
        self,
        records: Records,
        component: Component,
        experiment_settings: Dict[str, Any] = None,
    ) -> None:
        pass

    def flush(self) -> None:
        pass


NoopResultsWriter = _NoopResultsWriter()


class TensorBoardRecordInfo(TypedDict):
    name: str
    record: Record
    step: int


class TensorboardResultsWriter(ResultsWriter):
    def __init__(
        self, path: str, write_every: int = 1, tb_initial_step: int = 0
    ) -> None:
        self.path = path
        self.write_every = write_every
        self.summary_writer: Optional[tf.summary.SummaryWriter] = None
        self.tb_step = tb_initial_step

        self.updates_since_write = 0
        self.record_queue: List[TensorBoardRecordInfo] = []

        logging.info(f"Creating Tensorboard writer at path {self.path}")

    def add_result(
        self,
        records: Records,
        component: Component,
        experiment_settings: Dict[str, Any] = None,
    ) -> None:
        for name, record in records.items():
            if record.track_tensorboard:
                self.record_queue.append(
                    TensorBoardRecordInfo(name=name, record=record, step=self.tb_step)
                )

        self.tb_step += 1
        self.updates_since_write = (self.updates_since_write + 1) % self.write_every
        if self.updates_since_write == 0:
            self.flush()

    def flush(self) -> None:
        logging.debug("Flushing Tensorboard results writer")
        if self.summary_writer is None:
            logging.info("First flush, creating new summary writer")
            # Create it after we've queued up a bunch of records, so
            # we know how big to make the summary writer queue.
            # Will also autoflush every 2 minutes.
            self.summary_writer = tf.summary.create_file_writer(
                self.path, max_queue=len(self.record_queue)
            )

        with self.summary_writer.as_default():
            for tb_record_info in self.record_queue:
                record = tb_record_info["record"]
                if isinstance(record, TensorRecord):
                    tf.summary.histogram(
                        tb_record_info["name"],
                        record.value,
                        step=tb_record_info["step"],
                    )
                elif isinstance(record, ScalarRecord):
                    tf.summary.scalar(
                        tb_record_info["name"],
                        record.value,
                        step=tb_record_info["step"],
                    )
        self.summary_writer.flush()
