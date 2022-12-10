import logging
import os
from typing import Any, Dict, Generic, List, Sequence, TypeVar


import tensorflow as tf

from ..constants import RECORDS_SEP
from ..datasets import BatchedDataset, DataSampleValue
from ..utils import safe_open_dir, safe_create_file
from ..component import Component, Records
from .results_tracker import (
    ResultsTracker,
    StatsResultsTracker,
    NoopProgressBar,
    TqdmProgressBar,
    records_for_logging,
)
from .results_writer import (
    JSONResultsWriter,
    PlotterResultsWriter,
    TensorboardResultsWriter,
    NoopResultsWriter,
)
from .callback import ValidatorCallback

T = TypeVar("T")


class Validator(Generic[T]):
    def __init__(
        self,
        experiment_id: str,
        components: List[Component[T]],
        dataset: BatchedDataset,
        callbacks: Sequence[ValidatorCallback] = (),
        use_progress_bar=False,
        write_tensorboard_dir=None,
        write_json_dir=None,
        write_plots_dir=None,
        no_tb_during_validation=False,
        experiment_tags: List[Dict[str, Any]] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.experiment_tags = experiment_tags
        if self.experiment_tags is None:
            self.experiment_tags = [{} for _ in components]
        self.components = components
        self.dataset = dataset
        self.callbacks = callbacks
        self.use_progress_bar = use_progress_bar
        self.write_tensorboard_dir = write_tensorboard_dir
        self.write_json_dir = write_json_dir
        self.write_plots_dir = write_plots_dir
        self.no_tb_during_validation = no_tb_during_validation

    @staticmethod
    @tf.function
    def validate_step(
        x_batch: DataSampleValue, component: Component[DataSampleValue]
    ) -> Records:
        print(
            "Tracing validate_step"
        )  # This will only run when tensorflow traces the function.
        return component(x_batch)

    def validate_component(
        self,
        component: Component[T],
        experiment_num: int,
        experiment_settings: Dict[str, Any],
    ) -> ResultsTracker:
        data_cardinality = self.dataset.cardinality
        assert data_cardinality != tf.data.UNKNOWN_CARDINALITY
        progress_bar = (
            TqdmProgressBar(
                total=None
                if data_cardinality == tf.data.INFINITE_CARDINALITY
                else data_cardinality
            )
            if self.use_progress_bar
            else NoopProgressBar
        )
        results_tracker = StatsResultsTracker()
        callbacks = [c.fresh_clone() for c in self.callbacks]

        use_tb = (self.write_tensorboard_dir is not None) and (
            not self.no_tb_during_validation
        )
        logging.debug(f"Are we using tensorboard during validation: {use_tb}")

        tensorboard_writer = (
            NoopResultsWriter
            if not use_tb
            else TensorboardResultsWriter(
                safe_open_dir(
                    os.path.join(
                        self.write_tensorboard_dir,
                        RECORDS_SEP.join(
                            [str(self.experiment_id), str(experiment_num)]
                        ),
                    )
                ),
                write_every=50,
                tb_initial_step=0,
            )
        )

        for data_batch in self.dataset:
            records = self.validate_step(data_batch.value, component)
            records = results_tracker.update(records, steps=data_batch.batch_size)
            for callback in callbacks:
                callback_results = callback(records)
                records = callback_results["records"]
                stop = callback_results["stop"]
                if stop:
                    logging.info(
                        f"Validator callback {callback} reached its stop condition. Stopping validation."
                    )
                    break
            records = progress_bar.update(records)

            tensorboard_writer.add_result(
                records, component, experiment_settings=experiment_settings
            )
            if stop:
                break

        logging.info(f"Results are {records_for_logging(records)}")
        progress_bar.close()
        tensorboard_writer.flush()
        return results_tracker

    def run(self):
        tensorboard_writer = (
            NoopResultsWriter
            if self.write_tensorboard_dir is None
            else TensorboardResultsWriter(
                safe_open_dir(
                    os.path.join(self.write_tensorboard_dir, self.experiment_id)
                ),
                write_every=1,
                tb_initial_step=0,
            )
        )
        json_results_writer = (
            NoopResultsWriter
            if self.write_json_dir is None
            else JSONResultsWriter(
                path=safe_create_file(
                    os.path.join(self.write_json_dir, f"{self.experiment_id}.json")
                ),
                write_every=1,
            )
        )
        plot_results_writer = (
            NoopResultsWriter
            if self.write_plots_dir is None
            else PlotterResultsWriter(
                path=safe_create_file(
                    os.path.join(self.write_plots_dir, f"{self.experiment_id}.csv")
                ),
                write_every=1,
            )
        )
        for i, component in enumerate(self.components):
            logging.info(f"Running experiment {i+1}/{len(self.components)}")
            if self.experiment_tags is not None:
                experiment_settings = self.experiment_tags[i]
                logging.info(f"Experiment settings {experiment_settings}")
            else:
                experiment_settings = {}

            result_tracker = self.validate_component(
                component, experiment_num=i, experiment_settings=experiment_settings
            )
            results = result_tracker.results()

            json_results_writer.add_result(results, component, experiment_settings)
            tensorboard_writer.add_result(results, component, experiment_settings)
            plot_results_writer.add_result(results, component, experiment_settings)

    def serialize(self) -> Dict[str, Any]:
        raise NotImplementedError()
