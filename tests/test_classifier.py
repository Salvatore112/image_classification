import os
from pathlib import Path

import cv2
import pytest

from image_classification import PATH_TO_DEFAULT_MODULE
from image_classification.classifier import ImageClassifier


class TestUI:
    TEST_DATA_PATH = Path(__file__).parent / "test_dataset"
    MINIMAL_CONFIDENCE = 0.51

    @staticmethod
    @pytest.fixture()
    def classifier():
        _classifier = ImageClassifier()
        _classifier.restore(PATH_TO_DEFAULT_MODULE)
        return _classifier

    @staticmethod
    def get_jpg_files(directory):
        jpg_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(".jpg"):
                    jpg_files.append(cv2.imread(str(Path(root) / file)))
        return jpg_files

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param(
                "trash",
                id="trash",
            ),
            pytest.param(
                "glass",
                id="glass",
            ),
            pytest.param(
                "clothes",
                id="clothes",
            ),
            pytest.param(
                "metal",
                id="metal",
            ),
            pytest.param(
                "plastic",
                id="plastic",
            ),
            pytest.param(
                "cardboard",
                id="cardboard",
            ),
            pytest.param(
                "paper",
                id="paper",
            ),
            pytest.param(
                "biological",
                id="biological",
            ),
            pytest.param(
                "shoes",
                id="shoes",
            ),
        ],
    )
    def test_classes_prediction(self, name: str, classifier: ImageClassifier):
        images = self.get_jpg_files(self.TEST_DATA_PATH / name)
        for image in images:
            prediction = list(classifier.predict(image).items())
            ans = max(prediction, key=lambda x: x[1])
            assert ans[1] >= self.MINIMAL_CONFIDENCE
            assert ans[0] == name
