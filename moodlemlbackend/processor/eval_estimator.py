import os
from evalml.pipelines.pipeline_base import PipelineBase
from moodlemlbackend.model import EvalMlModel
from .base import BaseEstimater


class EvalClassifier(BaseEstimater):
    """EvalML based classifier"""

    def __init__(self, modelid, directory, dataset=None):
        super(EvalClassifier, self).__init__(modelid, directory, dataset=dataset)
        self.runner = None
        self.best_pipeline = None

    def get_classifier(self, X, y, initial_weights=False):
        """Gets the classifier"""
        self.runner = EvalMlModel()
        self.best_pipeline = self.runner.fit(X, y)
        return self.best_pipeline

    def load_classifier(self, model_dir=False):
        """Loads a previously stored classifier"""
        if model_dir is False:
            model_dir = self.persistencedir

        #classifier = super(EvalClassifier, self).load_classifier(model_dir)
        path = os.path.join(model_dir, 'model.ckpt')
        self.best_pipeline = PipelineBase.load(path)
        return self.best_pipeline

    def store_classifier(self, trained_classifier):
        """Stores the provided classifier"""
        path = os.path.join(self.persistencedir, 'model.ckpt')
        self.best_pipeline.save(path)
        return self.best_pipeline

    def export_classifier(self, exporttmpdir):
        # TODO implement this
        pass

    def import_classifier(self, importdir):
        # TODO implement this
        pass
