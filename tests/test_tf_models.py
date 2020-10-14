import numpy as np
from sklearn.datasets import load_breast_cancer
from moodlemlbackend.model.tensor import TF

def get_test_data():
    return load_breast_cancer()

def test_tf_model_init(tmp_path):
    """ Can we even init a model?"""
    data = get_test_data()
    # All columns but the last one.
    x = data.data
    y = data.target

    tf = TF(x.shape[0], len(np.unique(y)), 1000, 100, 0.5, tmp_path)
    assert tf is not None