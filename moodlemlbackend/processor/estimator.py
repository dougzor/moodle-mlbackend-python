"""Abstract estimator module, will contain just 1 class."""
import logging
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from moodlemlbackend.model import TFModel
from .base import BaseEstimater

logger = logging.getLogger(__name__)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

OK = 0
GENERAL_ERROR = 1
NO_DATASET = 2
LOW_SCORE = 4
NOT_ENOUGH_DATA = 8

PERSIST_FILENAME = 'classifier.pkl'
EXPORT_MODEL_FILENAME = 'model.json'


class Classifier(BaseEstimater):
    """Tensorflow Neural Net classifier"""

    def __init__(self, modelid, directory, dataset=None):
        super(Classifier, self).__init__(modelid, directory, dataset=dataset)

        self.tensor_logdir = self.get_tensor_logdir()
        if os.path.isdir(self.tensor_logdir) is False:
            if os.makedirs(self.tensor_logdir) is False:
                raise OSError('Directory ' + self.tensor_logdir +
                              ' can not be created.')

    def get_classifier(self, X, y, initial_weights=False):
        """Gets the classifier"""
        try:
            n_rows = X.shape[0]
        except AttributeError:
            # No X during model import.
            # n_rows value does not really matter during import.
            n_rows = 1

        if n_rows < 1000:
            batch_size = n_rows
        else:
            # A min batch size of 1000.
            x_tenpercent = int(n_rows / 10)
            batch_size = max(1000, x_tenpercent)

        # We need ~10,000 iterations so that the 0.5 learning rate decreases
        # to 0.01 with a decay rate of 0.96. We use 12,000 so that the
        # algorithm has some time to finish the training on lr < 0.01.
        starter_learning_rate = 0.5
        if n_rows > batch_size:
            n_epoch = int(12000 / (n_rows / batch_size))
        else:
            # Less than 1000 rows (1000 is the minimum batch size we defined).
            # We don't need to iterate than many times if we have less than
            # 1000 records, starting with 0.5 the learning rate will get to
            # ~0.05 in 5000 epochs.
            n_epoch = 5000

        n_classes = self.n_classes
        n_features = self.n_features

        model = TFModel(n_features, n_classes, n_epoch, batch_size,
                        starter_learning_rate, self.get_tensor_logdir(),
                        initial_weights=initial_weights)
        
        model.set_tensor_logdir(self.get_tensor_logdir())

        return model

    def store_classifier(self, trained_classifier):
        """Stores the classifier and saves a checkpoint of the tensors state"""

        # Store the graph state.
        saver = tf.train.Saver(save_relative_paths=True)
        sess = trained_classifier.get_session()

        path = os.path.join(self.persistencedir, 'model.ckpt')
        saver.save(sess, path)

        # Also save it to the logs dir to see the embeddings.
        path = os.path.join(self.get_tensor_logdir(), 'model.ckpt')
        saver.save(sess, path)

        # Save the class data.
        super(Classifier, self).store_classifier(trained_classifier)

    def load_classifier(self, model_dir=False):
        """Loads a previously trained classifier and restores its state"""

        if model_dir is False:
            model_dir = self.persistencedir

        classifier = super(Classifier, self).load_classifier(model_dir)

        classifier.set_tensor_logdir(self.get_tensor_logdir())

        # Now restore the graph state.
        saver = tf.train.Saver(save_relative_paths=True)
        path = os.path.join(model_dir, 'model.ckpt')
        saver.restore(classifier.get_session(), path)
        return classifier

    def export_classifier(self, exporttmpdir):
        if self.classifier_exists():
            classifier = self.load_classifier()
        else:
            return False

        export_vars = {}

        # Get all the variables in in initialise-vars scope.
        sess = classifier.get_session()
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope='initialise-vars'):
            # Converting to list as numpy arrays can't be serialised.
            export_vars[var.op.name] = var.eval(sess).tolist()

        # Append the number of features.
        export_vars['n_features'] = classifier.get_n_features()
        export_vars['n_classes'] = classifier.get_n_classes()

        vars_file_path = os.path.join(exporttmpdir, EXPORT_MODEL_FILENAME)
        with open(vars_file_path, 'w') as vars_file:
            json.dump(export_vars, vars_file)

        return exporttmpdir

    def import_classifier(self, importdir):

        model_vars_filepath = os.path.join(importdir,
                                           EXPORT_MODEL_FILENAME)

        with open(model_vars_filepath) as vars_file:
            import_vars = json.load(vars_file)

        self.n_features = import_vars['n_features']
        if "n_classes" in import_vars:
            self.n_classes = import_vars['n_classes']
        else:
            self.n_classes = 2

        classifier = self.get_classifier(False, False,
                                         initial_weights=import_vars)

        self.store_classifier(classifier)

    def get_tensor_logdir(self):
        """Returns the directory to store tensorflow framework logs"""
        return os.path.join(self.logsdir, 'tensor')
