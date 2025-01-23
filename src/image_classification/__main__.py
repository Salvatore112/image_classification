import argparse

from image_classification.classifier_util import CFG, ClassifierUtil, RunMode


def get_cfg():
    parser = argparse.ArgumentParser(description="Classifier parameters")
    CFG.add_arguments(parser)
    args = parser.parse_args()
    return CFG.get_config(args)


def main():
    cfg = get_cfg()
    classifier_util = ClassifierUtil(cfg)

    match cfg.mode:
        case RunMode.FIT:
            classifier_util.fit()
        case RunMode.PREDICT:
            res = classifier_util.predict()
            ClassifierUtil.render_predict_res(res)
        case _:
            raise ValueError(f"Invalid mode: {cfg.mode}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"Please check your configuration: {e}")
        exit(1)
    except Exception as e:
        print(f"Internal error: {e}")
        exit(2)
