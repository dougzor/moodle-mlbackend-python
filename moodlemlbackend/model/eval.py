import evalml
from .base import BaseModel


class EvalMlModel(BaseModel):
    def __init__(self):
        self.best_pipeline = None
        self.automl = None

    def build_graph(self, initial_weights=False):
        pass

    def fit(self, X, y):
        """Fits provided data into the session"""
        self.automl = evalml.AutoMLSearch(
            problem_type='binary',
            objective='accuracy binary',
            additional_objectives=['auc', 'balanced accuracy binary', 'precision'],
            max_iterations=10,
            optimize_thresholds=True
        )

        self.automl.search(X, y)
        self.best_pipeline = self.automl.best_pipeline
        self.best_pipeline.fit(X, y)
        return self.best_pipeline

    def predict(self, x):
        """Returns predictions"""
        return self.best_pipeline.predict(x)

    def predict_proba(self, x):
        """Returns predicted probabilities"""
        return self.best_pipeline.predict_proba(x)
