from moodlemlbackend.processor import EvalMLEstimator, TFEstimator
import os
import pytest


def get_test_data_path():
    return os.path.join(os.path.dirname(__file__), '../test_data/breast_cancer.csv')


testdata = [
    TFEstimator,
    EvalMLEstimator,
]

@pytest.mark.parametrize("estimator", testdata)
def test_estimator_setup(estimator, tmp_path):
    """Can the estimators be setup and trained"""
    test_data_file = get_test_data_path()
    classifier = estimator(1, tmp_path)
    result = classifier.train_dataset(test_data_file)
    assert result['status'] == 0


@pytest.mark.parametrize("estimator", testdata)
def test_estimator_storage(estimator, tmp_path):
    """Can the estimators be setup and trained"""
    test_data_file = get_test_data_path()
    classifier = estimator(1, tmp_path)
    result = classifier.train_dataset(test_data_file)
    assert result['status'] == 0

    loaded_classifier = classifier.load_classifier()

    assert type(classifier.trained_classifier) == type(loaded_classifier)
