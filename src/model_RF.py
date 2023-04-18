from sklearn.ensemble import RandomForestClassifier


class RFModel:

    def __init__(self, **args):
        """
        init RFModel (wrapper for sklearn.ensemble.RandomForestClassifier)

        :param args: **args for sklearn.ensemble.RandomForestClassifier
        """
        self._model = RandomForestClassifier(**args)

    def fit(self, X, y):
        """
        Method fit model

        :param X: features in sklearn format
        :param y: categories in sklearn format
        :return: <RFModel>
        """
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

