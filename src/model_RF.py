from sklearn.ensemble import RandomForestClassifier


class RFModel:

    def __init__(self, **args):
        self._model = RandomForestClassifier(**args)

    def fit(self, X, y):
        self._model = self._model.fit(X, y)
        return self

    def predict(self, X, y):
        return self._model.predict(X, y)

    def score(self, X, y):
        return self._model.score(X, y)

