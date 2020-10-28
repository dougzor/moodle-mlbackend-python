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
