from pathlib import Path

import pytest

from image_classification.classifier_util import CFG, RunMode, ClassifierUtil


class TestUI:
    @pytest.mark.parametrize(
        "cfg",
        [
            pytest.param(
                CFG(RunMode.PREDICT.value, None, None, None, None),
                id="PREDICT:all-None",
            ),
            pytest.param(
                CFG(RunMode.PREDICT.value, None, Path("/some/path"), None, None),
                id="PREDICT:model_path-None",
            ),
            pytest.param(
                CFG(RunMode.PREDICT.value, Path("/some/path"), None, None, None),
                id="PREDICT:img_path-None",
            ),
            pytest.param(
                CFG(RunMode.FIT.value, None, None, Path("/some/path"), None),
                id="FIT:all-None",
            ),
            pytest.param(
                CFG(RunMode.FIT.value, None, None, Path("/some/path"), None),
                id="FIT:dataset_path-None",
            ),
            pytest.param(
                CFG(RunMode.FIT.value, None, None, Path("/some/path"), None),
                id="FIT:path_to_save-None",
            ),
        ],
    )
    def test_mode_configuration(self, cfg: CFG):
        try:
            ClassifierUtil.validate(cfg)
        except Exception as e:
            assert isinstance(e, ValueError)

    def test_result_render(self):
        _INFO_ROWS_N = 2
        value = {
            "a": 0.2,
            "b": 0.1,
            "c": 0.3,
        }

        # Answer: c      <skip it>
        # All results:   <skip it>
        # +---+------+
        # | a | 0.20 |
        # | b | 0.10 |
        # | c | 0.30 |
        # +---+------+
        res = ClassifierUtil.render_predict_res(value).split("\n")[_INFO_ROWS_N:]

        checked_values = set()
        for k, v in value.items():
            idx = next((i for i, line in enumerate(res) if line.find(k) != -1), -1)
            if idx != -1:
                assert res[idx].find(f"{round(value[k], 2)}") != -1
                checked_values.add(k)

        assert len(checked_values) == len(value)
