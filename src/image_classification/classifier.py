from __future__ import annotations

import os
from abc import abstractmethod
from pathlib import Path
from typing import Protocol, Any

import cv2
import joblib
import numpy as np
import pandas as pd
from cv2 import Mat
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from image_classification import RANDOM_SEED, TEST_SIZE, MODEL_NAME
from image_classification.features import (
    IFeature,
    AmountOfYellow,
    AmountOfSilver,
    AmountOfLineCurvature,
    AmountOfParallelLines,
    AmountOfCylinders,
    AmountOfReflections,
    AmountOfTransparency,
    AmountOfTextureSmoothness,
    AmountOfTextureShininess,
    AmountOfSurfaceAnisotropy,
    AmountOfAspectRatio,
    AmountOfWhiteness,
)


class IClassifier(Protocol):
    @abstractmethod
    def predict(self, img: Mat | np.ndarray[Any, np.dtype]) -> dict[Any, float]: ...

    @abstractmethod
    def fit(self, train_dataset_path: Path): ...

    @abstractmethod
    def store(self, save_path: Path): ...

    @abstractmethod
    def restore(self, model_path: Path): ...


class ImageClassifier(IClassifier):
    _RANDOM_SEED = RANDOM_SEED
    _TEST_SIZE = TEST_SIZE
    _MODEl_NAME = MODEL_NAME

    features: list[IFeature] = [
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

    classes: list[str] = [
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

    _classifier: RandomForestClassifier | None

    def __init__(self, classifier: RandomForestClassifier | None = None):
        self._classifier = classifier

    @classmethod
    def _get_features(cls, img: np.ndarray) -> tuple[int, ...]:
        return tuple([feature(img) for feature in cls.features])

    @classmethod
    def _get_feature_names(cls) -> list[str]:
        return [feature.name() for feature in cls.features]

    @staticmethod
    def on_inited_classifier(func):
        def wrapper(self, *args, **kwargs):
            self: ImageClassifier = self
            assert self._classifier is not None
            return func(self, *args, **kwargs)

        return wrapper

    @on_inited_classifier
    def predict(self, img: Mat | np.ndarray[Any, np.dtype]) -> dict[Any, float]:
        features_list = self._get_features(img)
        features_df = pd.DataFrame([features_list], columns=self._get_feature_names())
        probabilities = self._classifier.predict_proba(features_df)
        return {
            class_name: prob
            for class_name, prob in zip(self._classifier.classes_, probabilities[0])
        }

    def fit(self, train_dataset_path: Path):
        features = []
        labels = []

        for class_name in self.classes:
            class_path = train_dataset_path / class_name
            image_names = os.listdir(class_path)
            train_images, test_images_class = train_test_split(
                image_names, test_size=TEST_SIZE, random_state=self._RANDOM_SEED
            )

            for image_name in train_images:
                image = cv2.imread(class_path / image_name)
                assert image is not None
                features_list = self._get_features(image)
                features.append(features_list)
                labels.append(class_name)

        df = pd.DataFrame(features, columns=self._get_feature_names())
        df["label"] = labels
        x, y = df[self._get_feature_names()], df["label"]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        self._classifier = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED
        )
        self._classifier.fit(x_train, y_train)

    @on_inited_classifier
    def store(self, save_path: Path):
        joblib.dump(self._classifier, save_path / f"{MODEL_NAME}.pkl")

    def restore(self, pkl_path: Path):
        self._classifier = joblib.load(pkl_path)
