from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from tabulate import tabulate

import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from image_classification import RANDOM_SEED, TEST_SIZE, MODEL_NAME
from image_classification.features import (
    Feature,
    AmountOfYellow,
    AmountOfSilver,
    AmountOfParallelLines,
    AmountOfCylinders,
    AmountOfReflections,
    AmountOfTransparency,
    AmountOfTextureSmoothness,
    AmountOfTextureShininess,
    AmountOfSurfaceAnisotropy,
    AmountOfAspectRatio,
    AmountOfWhiteness,
    AmountOfLineCurvature,
)


class RunMode(Enum):
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
    mode: RunMode
    img_path: Path | None
    model_path: Path | None
    dataset_path: Path | None
    path_to_save: Path | None

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
    features: list[Feature]

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
        self.classes = [
            "trash",
            "glass",
            "battery",
            "clothes",
            "metal",
            "plastic",
            "cardboard",
            "paper",
            "biological",
            "shoes",
        ]
        self.features = [
            AmountOfYellow(),
            AmountOfSilver(),
            AmountOfParallelLines(),
            AmountOfCylinders(),
            AmountOfReflections(),
            AmountOfTransparency(),
            AmountOfTextureSmoothness(),
            AmountOfTextureShininess(),
            AmountOfSurfaceAnisotropy(),
            AmountOfAspectRatio(),
            AmountOfWhiteness(),
            AmountOfLineCurvature(),
        ]

    def _get_features(self, img: np.ndarray) -> tuple[int, ...]:
        return tuple([feature(img) for feature in self.features])

    def _get_feature_names(self) -> list[str]:
        return [feature.name() for feature in self.features]

    def fit(self):
        features = []
        labels = []

        for class_name in self.classes:
            class_path = self.cfg.dataset_path / class_name
            image_names = os.listdir(class_path)
            train_images, test_images_class = train_test_split(
                image_names, test_size=TEST_SIZE, random_state=RANDOM_SEED
            )

            for image_name in train_images:
                image = cv2.imread(class_path / image_name)
                assert image is not None
                features_list = self._get_features(image)
                features.append(features_list)
                labels.append(class_name)

        # TODO: add ability to check tests

        df = pd.DataFrame(features, columns=self._get_feature_names())
        df["label"] = labels
        X = df[self._get_feature_names()]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        classifier.fit(X_train, y_train)
        joblib.dump(classifier, self.cfg.path_to_save / f"{MODEL_NAME}.pkl")

    def predict(self) -> dict[str, float]:
        classifier = joblib.load(self.cfg.model_path)
        image = cv2.imread(str(self.cfg.img_path))
        assert image is not None

        features_list = self._get_features(image)
        features_df = pd.DataFrame([features_list], columns=self._get_feature_names())
        probabilities = classifier.predict_proba(features_df)
        return {
            class_name: prob
            for class_name, prob in zip(classifier.classes_, probabilities[0])
        }

    @staticmethod
    def render_predict_res(pred_dict: dict[str, float]):

        raw_results = [(name, val) for name, val in pred_dict.items()]

        print(

        )

        print(
            f'Answer: {max(raw_results, key=lambda x: x[1])[0]}\n'
            f'All results:\n{tabulate(
                raw_results,
                floatfmt=".2f",
                showindex=False,
                tablefmt="psql",
            )}\n'
        )
