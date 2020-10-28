from .base import BaseModel

import evalml

class EvalMlModel(BaseModel):
    def __init__(self):
        self.best_pipeline = None

    def build_graph(self, initial_weights=False):
        pass

    def fit(self, X, y):
        """Fits provided data into the session"""
        automl = evalml.AutoMLSearch(
            problem_type='binary',
            objective='accuracy binary',
            additional_objectives=['auc', 'balanced accuracy binary', 'precision'],
            max_iterations=10,
            optimize_thresholds=True
        )

        automl.search(X, y)
        self.best_pipeline = automl.best_pipeline
        self.best_pipeline.fit(X, y)
        return self.best_pipeline

    def predict(self, x):
        """Returns predictions"""
        return self.best_pipeline.predict(x)

    def predict_proba(self, x):
        """Returns predicted probabilities"""
        return self.best_pipeline.predict_proba(x)
