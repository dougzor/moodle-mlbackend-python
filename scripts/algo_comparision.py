import numpy as np
from sklearn.datasets import load_breast_cancer
from moodlemlbackend.model.tensor import TF
import evalml
from moodlemlbackend.model.eval import EvalMlRunner
import tempfile
from sklearn.metrics import accuracy_score

def get_test_data():
    return load_breast_cancer()


def fit_tf_model(X_train, y_train, tmp_path):
    tf = TF(X_train.shape[1], len(np.unique(y_train)), 1000, 1000, 0.5, tmp_path)
    tf.fit(X_train.values, y_train.values)
    return tf
    

def fit_eval_model(X_train, y_train):
    runner = EvalMlRunner()
    best_pipeline = runner.fit(X_train, y_train)
    return best_pipeline


def evaluate_models():
    """ Can we even init a model?"""
    data = get_test_data()
    x = data.data
    y = data.target

    X_train, X_holdout, y_train, y_holdout = evalml.preprocessing.split_data(x, y, test_size=0.2, random_state=0)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tf = fit_tf_model(X_train, y_train, tmpdirname)

    eval_pipeline = fit_eval_model(X_train, y_train)

    tf_score = accuracy_score(y_holdout, tf.predict(X_holdout))
    evalml_score = accuracy_score(y_holdout, eval_pipeline.predict(X_holdout))

    print(f"Tensor Flow score: {tf_score}")
    print(f"EvalML score: {evalml_score}")


if __name__ == "__main__":
    evaluate_models()