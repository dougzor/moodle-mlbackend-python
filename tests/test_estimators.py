from moodlemlbackend.processor.estimator import Classifier
import os

def get_test_data_path():
    return os.path.join(os.path.dirname(__file__), '../test_data/breast_cancer.csv')

def test_tf_classifer(tmp_path):
    """ Can we even init a model?"""
    test_data_file = get_test_data_path()
    classifier = Classifier(1, tmp_path)
    result = classifier.train_dataset(test_data_file)
    assert result['status'] == 0