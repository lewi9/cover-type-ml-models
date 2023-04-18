from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


class HistGradientBoostingModel:

    def __init__(self, **args):
        """
        init HistGradientBoostingClassifier (wrapper for sklearn.ensemble.HistGradientBoostingClassifier)

        :param args: **args for sklearn.ensemble.HistGradientBoostingClassifier
        """
        self._model = HistGradientBoostingClassifier(**args)

    def fit(self, X, y):
        """
        Method fit model

        :param X: features in sklearn format
        :param y: categories in sklearn format
        :return: <HistGradientBoosingModel>
        """
        # weights = compute_sample_weight("balanced",y)
        # self._model = self._model.fit(X, y, weights)
        self._model = self._model.fit(X, y)
        return self

    def predict(self, X):
        """
        Method predict values

        :param X: features in sklearn format
        :return: categories in sklearn format - predicted values
        """
        return self._model.predict(X)

    def score(self, X, y):
        """
        Method calc mean accuracy of self.predict()

        :param X: features in sklearn format
        :param y: categories in sklearn format
        :return: <float> mean accuracy
        """
        return self._model.score(X, y)
