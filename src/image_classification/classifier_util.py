from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from tabulate import tabulate

import cv2

from image_classification import PATH_TO_DEFAULT_MODULE
from image_classification.classifier import IClassifier, ImageClassifier


class RunMode(Enum):
    """
    Enum representing the available modes of operation for the application.

    Attributes
    ----------
    FIT : str
        Represents the training mode, where a model is trained on a dataset.

    PREDICT : str
        Represents the prediction mode, where a trained model is used to make predictions.
    """

    FIT = "fit"
    PREDICT = "predict"

    @staticmethod
    def construct(arg: str):
        match arg:
            case RunMode.FIT.value:
                return RunMode.FIT
            case RunMode.PREDICT.value:
                return RunMode.PREDICT
        raise ValueError(f"Unexpected value {arg}")


@dataclass(frozen=True)
class CFG:
    """
    Configuration class for managing the parameters required to run the application.

    This class defines the structure of the configuration used for various modes of operation
    (e.g., training, prediction). It is immutable (`frozen=True`) and can be initialized
    either programmatically or via command-line arguments.

    Attributes
    ----------
    mode : RunMode
        The mode in which the application will run. Possible values are defined in the `RunMode` enum.
        Examples include `RunMode.FIT` for training and `RunMode.PREDICT` for inference.

    img_path : Path | None
        The path to the image used for prediction, if applicable. Optional and can be `None`
        if not required in the current mode.

    model_path : Path | None
        The path to the pre-trained model file. Required for modes that involve loading a model
        (e.g., `RunMode.PREDICT`).

    dataset_path : Path | None
        The path to the training dataset. Required for modes that involve training a model
        (e.g., `RunMode.FIT`).

    path_to_save : Path | None
        The path where the trained model or output should be saved. Defaults to
        `PATH_TO_DEFAULT_MODULE` if not explicitly provided.

    Methods
    -------
    add_arguments(parser: ArgumentParser) -> None
        Adds command-line argument definitions to the provided ArgumentParser instance.
        This method defines arguments corresponding to the attributes of the `CFG` class.

    get_config(args: Namespace) -> CFG
        Creates a `CFG` instance by extracting values from the given `argparse.Namespace`.
        This is typically used to convert parsed command-line arguments into a configuration object.

    Notes
    -----
    - This class is designed to work seamlessly with `argparse` for command-line argument parsing.
    - The `RunMode` enum should define the valid operational modes (`FIT`, `PREDICT`, etc.).
    - The `PATH_TO_DEFAULT_MODULE` serves as the default save path if `path_to_save` is not specified.
    """

    mode: RunMode
    img_path: Path | None
    model_path: Path | None
    dataset_path: Path | None
    path_to_save: Path | None = PATH_TO_DEFAULT_MODULE

    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument(
            "mode",
            type=str,
            choices=[str(RunMode.FIT.value), str(RunMode.PREDICT.value)],
            help="available modes",
        )
        parser.add_argument("-i", "--img_path", type=Path, help="path to image")
        parser.add_argument("-m", "--model_path", type=Path, help="path to model")
        parser.add_argument("-d", "--dataset_path", type=Path, help="path to dataset")
        parser.add_argument(
            "-s", "--path_to_save", type=Path, help="path to save model"
        )

    @staticmethod
    def get_config(args: Namespace) -> CFG:
        def _mb_path(value: str) -> Path | None:
            return Path(value) if value else None

        return CFG(
            mode=RunMode.construct(args.mode),
            img_path=_mb_path(args.img_path),
            model_path=_mb_path(args.model_path),
            dataset_path=_mb_path(args.dataset_path),
            path_to_save=_mb_path(args.path_to_save),
        )


class ClassifierUtil:
    cfg: CFG
    classifier: IClassifier

    @staticmethod
    def validate(cfg: CFG):
        def _fail_if_not(value: Any | None, msg: str):
            if value is None:
                raise ValueError(msg)

        match cfg.mode:
            case RunMode.FIT:
                _fail_if_not(cfg.dataset_path, "No dataset to train the model")
                _fail_if_not(cfg.path_to_save, "No path to save the model")
            case RunMode.PREDICT:
                _fail_if_not(cfg.img_path, "No image to classify")
                _fail_if_not(
                    cfg.model_path,
                    "The path to the trained classifier is not specified",
                )
            case _:
                raise ValueError(f"Invalid mode: {cfg.mode}")

    def __init__(self, run_cfg: CFG):
        self.validate(run_cfg)
        self.cfg = run_cfg
        self.classifier = ImageClassifier()

    def fit_action(self):
        self.classifier.fit(self.cfg.dataset_path)
        self.classifier.store(self.cfg.path_to_save)

    def predict_action(self) -> dict[str, float]:
        self.classifier.restore(self.cfg.model_path)

        image = cv2.imread(str(self.cfg.img_path))
        if image is None:
            raise ValueError(f"Invalid image: {self.cfg.img_path}")

        return self.classifier.predict(image)

    @staticmethod
    def render_predict_res(pred_dict: dict[str, float]) -> str:
        raw_results = [(name, val) for name, val in pred_dict.items()]
        return (
            f'Answer: {max(raw_results, key=lambda x: x[1])[0]}\n'
            f'All results:\n{tabulate(
                raw_results,
                floatfmt=".2f",
                showindex=False,
                tablefmt="psql",
            )}\n'
        )
