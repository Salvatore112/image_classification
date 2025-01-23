import os
from pathlib import Path

import pytest

from image_classification import PATH_TO_DEFAULT_MODULE, MODEL_NAME
from image_classification.classifier_util import RunMode, CFG, ClassifierUtil


class TestPipeline:
    TEST_DATA_PATH = Path(__file__).parent / "test_dataset"

    @pytest.mark.parametrize(
        "dataset_path",
        [
            pytest.param(
                Path(__file__).parent / "test_dataset" / "battery",
                id="PREDICT case",
            ),
        ],
    )
    def test_predict_configuration(self, dataset_path: Path):
        try:
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(".jpg"):
                        cfg = CFG(
                            mode=RunMode.PREDICT,
                            model_path=PATH_TO_DEFAULT_MODULE,
                            img_path=Path(root) / file,
                            dataset_path=None,
                            path_to_save=None,
                        )
                        ClassifierUtil(cfg).predict_action()
        except Exception:
            pytest.fail()

    @pytest.mark.parametrize(
        "dataset_path",
        [
            pytest.param(
                Path(__file__).parent / "test_dataset",
                id="FIT case",
            ),
        ],
    )
    def test_fit_pipeline(self, dataset_path: Path):
        try:
            cfg = CFG(
                mode=RunMode.FIT,
                model_path=None,
                img_path=None,
                dataset_path=dataset_path,
                path_to_save=dataset_path,
            )
            ClassifierUtil(cfg).fit_action()
        except Exception:
            pytest.fail()
        finally:
            fl = dataset_path / f"{MODEL_NAME}.pkl"
            if fl.exists() and fl.is_file():
                fl.unlink()
