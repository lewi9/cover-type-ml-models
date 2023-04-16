import numpy as np
import pandas as pd

class HeuristicModel:

    def __init__(self, n_classes = 4, n_iters = 10):
        self._n_classes = n_classes
        self._n_iters = n_iters

    def fit(self, X, y):
        self._bins = []
        X_new = X.copy()
        for i in range(len(X.columns)):
            X_new.loc[:,X_new.columns[i]], bins = pd.cut(X.iloc[:,i],
                                                         bins=self._n_classes,
                                                         labels = False, retbins=True)
            self._bins.append(bins)

        X_keys = pd.DataFrame(X_new.drop_duplicates())

        accuracy = 0
        for i in range(self._n_iters):
            X_keys["predict"] = np.random.choice(y.unique(), len(X_keys))
            X_predicted = pd.merge(
                X_new, X_keys,
                how = "left",
                left_on=[x for x in range(len(X.columns))],
                right_on=[x for x in range(len(X.columns))])
            diff = X_predicted.predict.to_numpy() == y.to_numpy()
            if np.sum(diff) > accuracy:
                accuracy = np.sum(diff)
                self._keys = X_keys.copy()

        return self

    def predict(self, X):
        X_new = X.copy()
        for i in range(len(X.columns)):
            X_new.loc[:,X_new.columns[i]] = pd.cut(X.iloc[:,i], bins=self._bins[i], labels = False)

        X_predicted = pd.merge(
                X_new, self._keys,
                how = "left",
                left_on=[x for x in range(len(X.columns))],
                right_on=[x for x in range(len(X.columns))])

        return X_predicted.iloc[:,-1].to_numpy()

    def score(self, X, y):
        return np.sum(self.predict(X) == y.to_numpy())/len(X)

 
        
        
