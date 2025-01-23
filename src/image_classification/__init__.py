from pathlib import Path


def hello() -> str:
    return "Hello from image-classification!"


RANDOM_SEED = 42
TEST_SIZE = 0.2
MODEL_NAME = "random_forest_classifier"
PATH_TO_DEFAULT_MODULE = Path(__file__).parent / "models" / "v1.default-classifier.pkl"
