import argparse

from image_classification.classifier_util import CFG, ClassifierUtil


def get_cfg():
    parser = argparse.ArgumentParser(description="Classifier parameters")
    CFG.add_arguments(parser)
    args = parser.parse_args()
    return CFG.get_config(args)


def main():
    classifier_util = ClassifierUtil(get_cfg())
    print(classifier_util.cfg)


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"Please check your configuration: {e}")
        exit(1)
    except Exception as e:
        print(f"Internal error: {e}")
        exit(2)
