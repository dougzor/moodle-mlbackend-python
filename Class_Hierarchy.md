# Class structure #
=======================================================

## Processor ##
This module contains the higher level interface for interacting with a machine learning model. The abstract base class in `processor/base.by`contains the following abstract methods that an specific instance must implemented:
* `get_classifier()`
* `load_classifier()`
* `store_classifier()`
* `export_classifier()`
* `import_classifier()`

Typical usage of any classifier is the following:
```
model_id = 1
model_directory  ""
classifier = EvalMLEstimator(model_id, model_directory)

training_data_file = ""
train_result = classifier.train_dataset(training_data_file)
assert result['status'] == 0

new_data_file = ""
predicted_result = classifier.predict_dataset(new_data_file)
# Something with your predicted results here
```

## Model ##
This module's contains the underlying implementation of the modeling against a particular algorithm and/or framework. The `base.py` contains the following:
```
class BaseModel(object):
    def __init__(self):
        pass

    def build_graph(self, initial_weights=False):
        raise NotImplementedError()

    def fit(self, X, y):
        """Fits provided data into the session"""
        raise NotImplementedError()

    def predict(self, x):
        """Returns predictions"""
        raise NotImplementedError()

    def predict_proba(self, x):
        """Returns predicted probabilities"""
        raise NotImplementedError()
```

Notably the `fit`, `predict` and `predict_prob` fit the typical signatures you'd expect to see from models. The TensorFlow and EvalML models implement these interfaces.