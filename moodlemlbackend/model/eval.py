import evalml
from evalml import AutoMLSearch


class EvalMlRunner(object):

    def __init__(self):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['x']
        del state['y_']
        del state['y']
        del state['probs']
        del state['train_step']
        del state['sess']

        del state['file_writer']
        del state['merged']

        # We also remove this as it depends on the run.
        del state['tensor_logdir']

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.build_graph()
        self.start_session()

    def set_tensor_logdir(self, tensor_logdir):
        """Sets tensorflow logs directory

        Needs to be set separately as it depends on the
        run, it can not be restored."""

        self.tensor_logdir = tensor_logdir
        try:
            self.file_writer
            self.merged
        except AttributeError:
            # Init logging if logging vars are not defined.
            self.init_logging()
    
    def build_graph(self, initial_weights=False):
        pass

    def start_session(self):
        """Starts the session"""

        self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def init_logging(self):
        """Starts logging the tensors state"""
        self.file_writer = tf.summary.FileWriter(
            self.tensor_logdir, self.sess.graph)
        self.merged = tf.summary.merge_all()

    def get_session(self):
        """Return the session"""
        return self.sess

    def get_n_features(self):
        """Return the number of features"""
        return self.n_features

    def get_n_classes(self):
        """Return the number of features"""
        return self.n_classes

    def fit(self, X, y):
        """Fits provided data into the session"""
        automl = AutoMLSearch(problem_type='binary',
                              objective='accuracy binary',
                              additional_objectives=['auc', 'balanced accuracy binary', 'precision'],
                              max_iterations=10,
                              optimize_thresholds=True)

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