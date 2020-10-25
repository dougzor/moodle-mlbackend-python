from .estimator import Estimator
from evalml.pipelines.pipeline_base import PipelineBase
from moodlemlbackend.model.eval import EvalMlRunner
import os

class EvalClassifier(Estimator):
    """EvalML based classifier"""

    def get_classifier(self, X, y, initial_weights=False):
        """Gets the classifier"""
        runner = EvalMlRunner()
        self.best_pipeline = runner.fit(X, y)
        return self.best_pipeline

    def load_classifier(self, model_dir=False):
        """Loads a previously stored classifier"""
        if model_dir is False:
            model_dir = self.persistencedir

        classifier = super(EvalClassifier, self).load_classifier(model_dir)
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
