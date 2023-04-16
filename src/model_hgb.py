from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

class HistGradientBoostingModel:

    def __init__(self, **args):
        self._model = HistGradientBoostingClassifier(**args)

    def fit(self, X, y):
        #weights = compute_sample_weight("balanced",y)
        #self._model = self._model.fit(X, y, weights)
        self._model = self._model.fit(X, y)
        return self

    def predict(self, X, y):
        return self._model.predict(X, y)

    def score(self, X, y):
        return self._model.score(X, y)
