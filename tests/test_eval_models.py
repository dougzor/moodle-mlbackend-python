import numpy as np
import evalml
from sklearn.datasets import load_breast_cancer
from moodlemlbackend.model.eval import EvalMlRunner

def get_test_data():
    return load_breast_cancer()


def test_it():
    runner = EvalMlRunner()

    data = get_test_data()
    x = data.data
    y = data.target

    X_train, X_holdout, y_train, y_holdout = evalml.preprocessing.split_data(x, y, test_size=0.2, random_state=0)
    
    best_pipeline = runner.fit(X_train, y_train)
    score = best_pipeline.score(X_holdout, y_holdout, objectives=["accuracy binary"])

    print(score["Accuracy Binary"])

    assert score["Accuracy Binary"] > 0
