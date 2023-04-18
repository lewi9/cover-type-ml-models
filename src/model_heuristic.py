import numpy as np
import pandas as pd

class HeuristicModel:

    def __init__(self, n_classes = 4, n_iters = 10):
        """
        Create Heuristc model, based on data abstract to n_classes,
        assign random class to each unique configuration,
        iterate n_iters times to find the best solution.

        :param n_classes: <int> data abstract level
        :param n_iters: <int> number of iterations to find best solution
        """
        self._keys = None
        self._n_classes = n_classes
        self._n_iters = n_iters

    def fit(self, X, y):
        """
        Method fit model

        :param X: features in sklearn format
        :param y: categories in sklearn format
        :return: <HeuristicModel>
        """
        self._bins = []
        X_new = X.copy()
        for i in range(len(X.columns)):
            X_new.loc[:,X_new.columns[i]], bins = pd.cut(X.iloc[:,i], bins=self._n_classes,
                                                         labels = False, retbins=True)
            self._bins.append(bins)

        X_keys = pd.DataFrame(X_new.drop_duplicates())

        accuracy = 0
        for i in range(self._n_iters):
            X_keys["predict"] = np.random.choice(y.unique(), len(X_keys))
            X_predicted = pd.merge(X_new, X_keys, how = "left",
                                   left_on=[x for x in range(len(X.columns))],
                                   right_on=[x for x in range(len(X.columns))])
            diff = X_predicted.predict.to_numpy() == y.to_numpy()
            if np.sum(diff) > accuracy:
                accuracy = np.sum(diff)
                self._keys = X_keys.copy()

        return self

    def predict(self, X):
        """
        Method predict values

        :param X: features in sklearn format
        :return: categories in sklearn format - predicted values
        """
        X_new = X.copy()
        for i in range(len(X.columns)):
            X_new.loc[:,X_new.columns[i]] = pd.cut(X.iloc[:,i], bins=self._bins[i], labels = False)

        X_predicted = pd.merge(X_new, self._keys, how="left",
                               left_on=[x for x in range(len(X.columns))],
                               right_on=[x for x in range(len(X.columns))])
        return X_predicted.iloc[:, -1].fillna(np.random.choice(self._keys.iloc[:, -1].unique())).to_numpy()

    def score(self, X, y):
        """
        Method calc mean accuracy of self.predict()

        :param X: features in sklearn format
        :param y: categories in sklearn format
        :return: <float> mean accuracy
        """
        return np.sum(self.predict(X) == y.to_numpy())/len(X)

 
        
        
